from enum import Enum
from typing import Dict
import torch
import torch.fx as fx
from typing import Dict, Any
import matplotlib
matplotlib.use("Agg")        # no display needed; saves to file
import matplotlib.pyplot as plt
import numpy as np


class OP(str, Enum):
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"


class NodeType(Enum):
    """
    NodeType is a enum that records the type of the tensors in the graph.
    """

    PARAM = 0
    ACT = 1
    GRAD = 2
    OTHER = 3


# This is an example graph_profiler that extends the fx.Interpreter class, it
# will perform graph execution by running the graph node by node.


class GraphProfiler(fx.Interpreter):
    def __init__(self, module: fx.GraphModule, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)

        # You should perform the static analysis of the graph here. In
        # particular you might want to find the intermediate
        # nodes/activations/feature_maps in the graph that will be defined as
        # those nodes which are not parameters (not placeholder node types) but
        # are created during the forward pass and are also used in the backward
        # pass for computation.

    
        # important to note that each node object has node.name = unique string identifier
        # node.op = kind of operations (placeholder, call_function, output etc)
        #node.target = what the node calls 
        #node.all_input_nodes = list of all nodes whose outputs feed into this node (its dependencies)
        #node.users = dict of nodes that consume this node's output (who depends on it)

        # The boundary between the forward pass and backward pass can be
        # identified by locating the node '%sep : [num_users=1] =
        # call_function[target=torch.ops.separator.sep.default]' which will
        # define the end of the forward pass. You will see the loss function
        # after thsi operation and then you will encounter a node named,
        # '%sep_backward : [num_users=1] =
        # call_function[target=torch.ops.separator.sep_backward.default]'. This
        # node marks the beginning of the backward pass.
        self.sep_node = None
        self.sep_backward_node = None

        for node in self.module.graph.nodes: #iterate over every node in the FX computation
            #graph in topological order (sorted order)
            
            #here we are looking for the two boundary markers, separators, which tell
            #us where the forward pass ends and backward pass begins. will use them
            #later to split nodes into regions (forward and backward)
            
            
            if node.op == OP.CALL_FUNCTION and node.target == torch.ops.separator.sep.default:
                self.sep_node = node #the node where forward ends
            if node.op == OP.CALL_FUNCTION and node.target == torch.ops.separator.sep_backward.default:
                self.sep_backward_node = node #the node where backward begins


        #SPLIT UP GRAPH NODES INTO BELONGING TO FORWARD OR BACKWARD REGION
        #again walk through graph in topological order  and flip a state 
        #variable when hitting the boundary separatro nodes
        self.forward_nodes = set() 
        self.backward_nodes = set()
        region = "forward"
        for node in self.module.graph.nodes:
            if region == "forward":
                self.forward_nodes.add(node.name)
                if node is self.sep_node:
                    region = "between" #between loss and sep_backward
            elif region == "between":
                if node is self.sep_backward_node: 
                    region = "backward"
                    self.backward_nodes.add(node.name)
            else:
                self.backward_nodes.add(node.name)
            

         # The parameters of the models are the placeholder (input) nodes of the
        # graph. Note that not all the placeholder nodes of the graph are
        # parameters. The optimizer's states and the input mini-batch are also
        # placeholder nodes that given as inputs to the graph.

        # The parameters and gradients of the model can be otained using the
        # optimizer node's arguments. The optimizer node can be identified by
        # the node '%_fused_adam : [num_users=3] =
        # call_function[target=torch.ops.aten._fused_adam.default]'.
        # The argument at position 0 is the list of parameter nodes, while the
        # argument at position 1 is the list of gradient nodes.



        #IDENTIFYING PARAMS AND GRADIENTS USING THE OPTIMIZER NODES, "fused_adam" call

        #so we canfigure out which placeholdder nodes are model parameters and which are gradients
        #we can't tell from placeholder nodes alone, graph has many placeholders (params, gradients, optimizers etc)
        #and they all look the same

        #building lookup sets; sets of name strings used for fast membership checking.
        self.param_nodes = set()
        self.grad_nodes = set()

        optimizer_node = None
        #find the optimizer nodes by going through the list and searching for call function + fused adam call
        for node in self.module.graph.nodes:
            #fused_adam optimizer note takes its inputs in a fixed order
            #args[0] = list of parameter nodes, args[1] = list of gradient nodes, args[2+] = optimizer states
            if node.op == OP.CALL_FUNCTION and "fused_adam" in str(node.target):
                optimizer_node = node
                break
            #by finding the optimizer node and reading the args, we get ground-truth list
            #of which nodes are params and which are grads
        
        if optimizer_node is not None:
            args = optimizer_node.args
        
        #extracting param and grad names from the arguments -- lots of help from claude here
        #we do this b/c later when checking if node.name in self.param_nodes during classifcation,
        #comparing strings is simpler and avoids holding references to node objects
            if len(args) > 0 and isinstance(args[0], (list, tuple)):
                for p in args[0]:
                    if isinstance(p, fx.Node):
                        self.param_nodes.add(p.name)
            if len(args) > 1 and isinstance(args[1], (list, tuple)):
                for g in args[1]:
                    if isinstance(g, fx.Node):
                        self.grad_nodes.add(g.name)


        #FINDING THE ACTIVATIONS
        #key here is activation is any computed non-placehokder node in the forward region 
        #that is also used by at least one node in the backward region

        self.activation_nodes = set() #set b/c each of these nodes will be unique

        for node in self.module.graph.nodes:
            if node.name not in self.forward_nodes:
                continue

            if node.op == OP.PLACEHOLDER:
                continue
            if node.op == OP.OUTPUT:
                continue
            #check if any user of node is backward
            for user in node.users:
                if user.name in self.backward_nodes:
                    self.activation_nodes.add(node.name)
                    break


        #CLASSIFY EVERY NODE 
        self.node_types = {} #node.name = NodeType

        for node in self.module.graph.nodes:
            if node.name in self.param_nodes:
                self.node_types[node.name] = NodeType.PARAM
            elif node.name in self.grad_nodes:
                self.node_types[node.name] = NodeType.GRAD
            elif node.name in self.activation_nodes:  
                self.node_types[node.name] = NodeType.ACT
            else:
                self.node_types[node.name] = NodeType.OTHER            

        #STATIC ANALYSSIS WHICH IS FIRST AND LAST USE
        # For these intermediate nodes in the graph, you will record their last
        # use in the forward pass and their first use in the backward pass.
        
        #static analysis tells us how long the activation sits idle in memory

        self.last_forward_use = {a: None for a in self.activation_nodes} 
        #example: last_forward_use = {"relu_1": "linear_2"}
        self.first_backward_use = {a: None for a in self.activation_nodes}
        #exampl;e: first_backward_use = {"relu_1": "mm_backward_1"}
        
        #BOTH DICTS HELP ANSWER THE QUESTION HOW LONG DOES EACH ACTIVATION SIT IDLE IN GPU MEMORY
        #gap between last_forward_use and first_backward_use is wasted memory.



        for node in self.module.graph.nodes:
            for inp in node.all_input_nodes:
                if inp.name not in self.activation_nodes:
                    continue
                if node.name in self.forward_nodes:
                    self.last_forward_use[inp.name] = node.name   # keep overwriting — last one wins
                elif node.name in self.backward_nodes:
                    if self.first_backward_use[inp.name] is None:  # only record first
                        self.first_backward_use[inp.name] = node.name

        #DYNAMIC PROFILING
        self.runtimes_ms  = {n.name: [] for n in self.module.graph.nodes}
        self.mem_before   = {n.name: [] for n in self.module.graph.nodes}
        self.mem_after    = {n.name: [] for n in self.module.graph.nodes}
        self.output_bytes = {n.name: [] for n in self.module.graph.nodes}

        #AGGREGATE STATS
        self.avg_runtime_ms    = {}
        self.avg_mem_before_mb = {}
        self.avg_mem_after_mb  = {}
        self.avg_output_bytes  = {}


    def run(
        self,
        *args,
        initial_env: Dict[fx.Node, Any] | None = None,
        enable_io_processing: bool = True
    ) -> Any:
        return super().run(
            *args, initial_env=initial_env, enable_io_processing=enable_io_processing
        )

    def run_node(self, n: fx.Node) -> Any:
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()

        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()

        result = super().run_node(n)

        end.record()
        torch.cuda.synchronize()

        self.runtimes_ms[n.name].append(start.elapsed_time(end))
        self.mem_before[n.name].append(mem_before)
        self.mem_after[n.name].append(torch.cuda.memory_allocated())

        # record output tensor size
        if isinstance(result, torch.Tensor):
            self.output_bytes[n.name].append(result.nbytes)
        elif isinstance(result, (list, tuple)):
            self.output_bytes[n.name].append(
                sum(r.nbytes for r in result if isinstance(r, torch.Tensor))
            )
        else:
            self.output_bytes[n.name].append(0)

        return result


        

    def aggregate_stats(self) -> None:
        for node in self.module.graph.nodes:
            name = node.name
            n = len(self.runtimes_ms[name])
            if n == 0:
                continue
            self.avg_runtime_ms[name]    = sum(self.runtimes_ms[name]) / n
            # Convert bytes -> MB for readability
            self.avg_mem_before_mb[name] = sum(self.mem_before[name]) / n / (1024**2)
            self.avg_mem_after_mb[name]  = sum(self.mem_after[name])  / n / (1024**2)
            self.avg_output_bytes[name]  = sum(self.output_bytes[name]) / n

    def print_stats(self) -> None:
        # ── Per-operator table ────────────────────────────────────────────────
        print("\n" + "="*100)
        print("PER-OPERATOR STATISTICS")
        print("="*100)
        print(f"{'Node':<45} {'Type':<10} {'Runtime(ms)':<14} {'Mem Before(MB)':<18} {'Mem After(MB)'}")
        print("-"*100)

        total_ms = 0.0
        for node in self.module.graph.nodes:
            name  = node.name
            ntype = self.node_types.get(name, NodeType.OTHER).name
            rt    = self.avg_runtime_ms.get(name, 0.0)
            mb_b  = self.avg_mem_before_mb.get(name, 0.0)
            mb_a  = self.avg_mem_after_mb.get(name, 0.0)
            total_ms += rt
            print(f"{name:<45} {ntype:<10} {rt:<14.4f} {mb_b:<18.2f} {mb_a:.2f}")

        print(f"\nTotal iteration runtime: {total_ms:.2f} ms")

        # ── Activation static analysis table ─────────────────────────────────
        print("\n" + "="*100)
        print("ACTIVATION STATIC ANALYSIS")
        print("="*100)
        print(f"{'Activation':<45} {'Last Fwd Use':<40} {'First Bwd Use'}")
        print("-"*110)

        for act in sorted(self.activation_nodes):
            lf = self.last_forward_use.get(act)  or "N/A"
            fb = self.first_backward_use.get(act) or "N/A"
            print(f"{act:<45} {lf:<40} {fb}")

        total_act_mb = sum(
            self.avg_output_bytes.get(a, 0) for a in self.activation_nodes
        ) / (1024**2)
        print(f"\nTotal activations: {len(self.activation_nodes)}")
        print(f"Total activation memory: {total_act_mb:.2f} MB")

        # ── Memory by type summary ────────────────────────────────────────────
        print("\n" + "="*50)
        print("MEMORY BY TENSOR TYPE (output bytes summed)")
        print("="*50)
        type_mb = {t: 0.0 for t in NodeType}
        for node in self.module.graph.nodes:
            ntype = self.node_types.get(node.name, NodeType.OTHER)
            type_mb[ntype] += self.avg_output_bytes.get(node.name, 0.0) / (1024**2)
        for ntype, mb in type_mb.items():
            print(f"  {ntype.name:<10}: {mb:.2f} MB")

        # ── Plot ──────────────────────────────────────────────────────────────
        self._plot_memory_breakdown()
    
    def _plot_memory_breakdown(self) -> None:
        

        # Only plot nodes that we actually measured
        measured = [n for n in self.module.graph.nodes if n.name in self.avg_mem_after_mb]
        if not measured:
            return

        names      = [n.name for n in measured]
        total_mem  = [self.avg_mem_after_mb[n] for n in names]
        x          = np.arange(len(names))

        # ── Reconstruct a "live tensor" timeline ─────────────────────────────
        # For each node, its output tensor lives from when it's created (step i)
        # until its last user runs (step j). We use this to build a stacked area.
        node_idx = {n: i for i, n in enumerate(names)}

        # freed_at[name] = index of the last user node that we measured
        freed_at = {}
        for node in measured:
            last_user_step = node_idx[node.name]
            for user in node.users:
                if user.name in node_idx:
                    last_user_step = max(last_user_step, node_idx[user.name])
            freed_at[node.name] = last_user_step

        # type_mem[NodeType][step] = MB of live tensors of that type at this step
        type_mem = {t: np.zeros(len(names)) for t in NodeType}
        for node in measured:
            created = node_idx[node.name]
            freed   = freed_at[node.name]
            ntype   = self.node_types.get(node.name, NodeType.OTHER)
            out_mb  = self.avg_output_bytes.get(node.name, 0.0) / (1024**2)
            type_mem[ntype][created:freed+1] += out_mb

        # ── Plot 1: total memory trace ────────────────────────────────────────
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(18, 11))
        fig.suptitle("GPU Memory During Training Iteration", fontsize=13)

        ax0.plot(x, total_mem, color="steelblue", linewidth=1.5)
        if self.sep_node and self.sep_node.name in node_idx:
            ax0.axvline(node_idx[self.sep_node.name], color="red",
                        linestyle="--", label="sep (fwd end)")
        if self.sep_backward_node and self.sep_backward_node.name in node_idx:
            ax0.axvline(node_idx[self.sep_backward_node.name], color="orange",
                        linestyle="--", label="sep_backward (bwd start)")
        peak_idx = int(np.argmax(total_mem))
        ax0.annotate(f"Peak: {total_mem[peak_idx]:.1f} MB",
                    xy=(peak_idx, total_mem[peak_idx]),
                    xytext=(peak_idx + max(1, len(x)//15), total_mem[peak_idx]),
                    arrowprops=dict(arrowstyle="->"))
        ax0.set_title("Total GPU Memory Allocated (per operation)")
        ax0.set_xlabel("Operation Index (topological order)")
        ax0.set_ylabel("Memory (MB)")
        ax0.legend(); ax0.grid(alpha=0.3)

        # ── Plot 2: stacked area by tensor type ───────────────────────────────
        colors = {NodeType.PARAM: "green", NodeType.ACT: "steelblue",
                NodeType.GRAD: "red",   NodeType.OTHER: "gray"}
        labels = {NodeType.PARAM: "Parameters", NodeType.ACT: "Activations",
                NodeType.GRAD:  "Gradients",  NodeType.OTHER: "Other"}

        bottom = np.zeros(len(x))
        for ntype in [NodeType.PARAM, NodeType.ACT, NodeType.GRAD, NodeType.OTHER]:
            vals = type_mem[ntype]
            ax1.fill_between(x, bottom, bottom + vals,
                            label=labels[ntype], color=colors[ntype], alpha=0.75)
            bottom += vals

        if self.sep_node and self.sep_node.name in node_idx:
            ax1.axvline(node_idx[self.sep_node.name], color="red",
                        linestyle="--", label="sep (fwd end)")
        if self.sep_backward_node and self.sep_backward_node.name in node_idx:
            ax1.axvline(node_idx[self.sep_backward_node.name], color="orange",
                        linestyle="--", label="sep_backward (bwd start)")

        ax1.set_title("Live-Tensor Memory by Category")
        ax1.set_xlabel("Operation Index (topological order)")
        ax1.set_ylabel("Memory (MB)")
        ax1.legend(loc="upper left"); ax1.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("peak_memory_breakdown.png", dpi=150, bbox_inches="tight")
        print("\nSaved: peak_memory_breakdown.png")
        plt.close(fig)

    def reset_stats(self) -> None:
        for node in self.module.graph.nodes:
            self.runtimes_ms[node.name].clear()
            self.mem_before[node.name].clear()
            self.mem_after[node.name].clear()
            self.output_bytes[node.name].clear()
        self.avg_runtime_ms.clear()
        self.avg_mem_before_mb.clear()
        self.avg_mem_after_mb.clear()
        self.avg_output_bytes.clear()

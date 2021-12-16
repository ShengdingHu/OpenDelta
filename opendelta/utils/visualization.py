from logging import root
from rich.tree import Tree as RichTree
from rich import print as richprint
import torch
import torch.nn as nn
import re
from collections import OrderedDict


class Visualization(object):
    r"""Todo: 1. support distinguish delta model
              2. support visualize shared weight
    """
    def __init__(self, plm):
        self.plm = plm
        self.type_color = "green"
        self.param_color = "cyan"
        self.duplicate_color = "red"
        self.normal_color = "white"

 
    def structure_graph(self, rootname="root", expand_params=False, keep_non_params=False):
        self.keep_non_params = keep_non_params
        self.expand_params = expand_params

        module_ref_stack = [self.plm]
        root_tree = RichTree(rootname)
        tree_stack = [root_tree]
        while(len(module_ref_stack)>0):
            c_module = module_ref_stack.pop(0)
            tree = tree_stack.pop(0)
            for n,m in c_module.named_children():
                module_ref_stack.append(m)
                type_info = re.search(r'(?<=\').*(?=\')', str(type(m))).group()
                type_info = type_info.split(".")[-1]
                node_string = n+f"[{self.type_color}]({type_info})"
                newnode = tree.add(node_string)
                tree_stack.append(newnode)
                self.add_param_info_node(m, newnode)
          
        self.prune_tree(root_tree)
        if not self.expand_params:
            self.fold_param_node(root_tree)
        richprint(root_tree)
    
    def fold_param_node(self, t: RichTree, p:RichTree=None):
        if hasattr(t,"is_param_node") and t.is_param_node:
            p.label += " "+t.label
            return True # indicate whether should be removed
        elif len(t.children) == 0:
            if self.keep_non_params:
                return False
            else:
                return True
        else:
            rm_idx = []
            for idx, c in enumerate(t.children):
                if self.fold_param_node(t=c, p=t):
                    rm_idx.append(idx)
            t.children = [t.children[i] for i in range(len(t.children)) if i not in rm_idx]
            return False

    def prune_tree(self, t: RichTree):
        if len(t.children) == 0:
            setattr(t, "_finger_print", t.label)
            return

        for idx, sub_tree in enumerate(t.children):
            self.prune_tree(sub_tree)

        t_finger_print = t.label +"::"+";".join([x._finger_print for x in t.children])
        setattr(t, "_finger_print", t_finger_print)
        
        nohead_finger_print_dict = OrderedDict()
        for child_id, sub_tree in enumerate(t.children):
            fname_list = sub_tree._finger_print.split("::")
            if len(fname_list)==1:
                fname = fname_list[0]
            else:
                fname = "::".join(fname_list[1:])
            if fname not in nohead_finger_print_dict:
                nohead_finger_print_dict[fname] = [child_id]
            else:
                nohead_finger_print_dict[fname].append(child_id)

        new_childrens = []
        for groupname in nohead_finger_print_dict:
            representative_id = nohead_finger_print_dict[groupname][0]
            representative = t.children[representative_id]
            new_label = [t.children[idx].label for idx in nohead_finger_print_dict[groupname]]
            try:
                representative.label = self.extract_common_and_join(new_label)
            except:
                representative.label = ",".join(new_label)
            new_childrens.append(representative)
        t.children = new_childrens


    def extract_common_and_join(self, l):
        if len(l)==1:
            return l[0]
        commons = [x.strip(")").split(f"[{self.type_color}](") for x in l]
        type_hint_dict = OrderedDict()
        for x, y in commons:
            if y not in type_hint_dict:
                type_hint_dict[y] = [x]
            else:
                type_hint_dict[y].append(x)
        
        s = ""
        for t in type_hint_dict:
            group_components = type_hint_dict[t]
            group_components = self.neat_expr(group_components)
            s += f"[{self.duplicate_color}]{group_components}[{self.type_color}]({t})"
            s += f"[{self.normal_color}],"
        s = s[:-len(f"[{self.normal_color}],")]
        return s

    def neat_expr(self, l):
        try:
            s = self.ranges([int(x) for x in l])
            s = [str(x)+"-"+str(y) for x,y in s]
            return ",".join(s)
        except:
            return ",".join(l)
    
    def ranges(self,nums):
        nums = sorted(set(nums))
        gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
        edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
        return list(zip(edges, edges))

    def flattened_structure(self):
        for n, _ in self.plm.named_modules():
            print(n)

    def add_param_info_node(self, m, tree):
        known_module = [n for n,c in m.named_children()]
        for n,p in m.named_parameters():
            if n.split(".")[0] not in known_module:
                if len(n.split(".")) > 1: raise RuntimeError(f"The name field {n} should be a parameter since it doesn't appear in named_children, but it contains '.'")
                info = "[{}]{}:{}".format(self.param_color, n, list(p.shape) )
                new_node = tree.add(info)
                setattr(new_node, "is_param_node", True)

    
 
        
    


if __name__=="__main__":
    from openprompt.plms import load_plm
    import argparse
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model", type=str, default='t5-lm', help="We test both t5 and t5-lm in this scripts, the corresponding tokenizerwrapper will be automatically loaded.")
    parser.add_argument("--model_name_or_path", default="t5-large-lm-adapt")
    parser.add_argument("--cache_base", default='/home/hushengding/plm_cache/')
    parser.add_argument("--keep_non_params", action="store_true")
    parser.add_argument("--expand_params", action="store_true")
    args = parser.parse_args()
    plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.cache_base+args.model_name_or_path)

    visobj = Visualization(plm)
    visobj.structure_graph(rootname=args.model_name_or_path, keep_non_params=args.keep_non_params, expand_params=args.expand_params)

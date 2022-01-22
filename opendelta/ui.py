import sys
sys.path.append(".")
sys.path.append("..")
from opendelta.utils.visualization import Visualization
import web
import re, copy

from transformers import GPT2Model as Model
model = Model.from_pretrained("gpt2")
vis = Visualization(model)
tree = vis.structure_graph()

urls = (
    '/submit/(.*)', 'submit',
    '/(.*)', 'hello',
)
app = web.application(urls, globals())
render = web.template.render('web/templates/')

space = "&nbsp;"
prefix0 = space * 9
prefix1 = f"│"+space*5
prefix2 = f"├─{space}"
prefix3 = f"└─{space}"

def colorfy(label):
    i = 0
    res = ""
    while i < len(label):
        if label[i] == '[':
            color = ""
            i += 1
            while label[i] != ']':
                color += label[i]
                i += 1
            i += 1
            if color[0].isdigit(): # dims but not color
                res += f'[{color}]'
            else:
                if res != "": res += '</span>'
                res += f'<span style="color: {color}">'
        else:
            res += label[i]
            i += 1
    res += '</span>'
    return res

compressed_pattern_1 = re.compile("[0-9]+-[0-9]+")
compressed_pattern_2 = re.compile(".+(,.+)+")

def expand_part(part):
    res = []
    if compressed_pattern_1.fullmatch(part):
        st, ed = map(int, part.split('-'))
        for i in range(st, ed+1):
            res.append( str(i) )
    elif compressed_pattern_2.fullmatch(part):
        for c in part.split(','):
            res.append( c )
    else:
        res.append( part )
    return res

def dfs(o, depth, last, old_name):
    html = ""
    module_names = expand_part(o.module_name)
    if depth > 0:
        old_last_1 = last[-1]
    if len(module_names) > 1:
        module_names = [o.module_name] + module_names
    for ith, module_name in enumerate(module_names):
        if ith == 0:
            html += f'<div>'
        elif ith == 1:
            html += f'<div class="expandable-sibling">'

        for i in range(depth-1):
            html += prefix0 if last[i] else prefix1
        if depth > 0:
            last[-1] = old_last_1 & (ith == 0 or ith == len(module_names)-1)
            html += prefix3 if last[-1] else prefix2
        length = len(o.children)
        if length > 0:
            html += f'<button class="collapsible button_inline">[+]</button>'
        name = old_name + module_name
        if ith > 0:
            label = f'[red]{module_name}{o.label[o.label.index("[",1):]}'
        else:
            label = o.label
        html += f'<button class="selectable button_inline" id="{name}">{colorfy(label)}</button>'
        if len(module_names) > 1 and ith == 0:
            html += '<button class="expandable button_inline">[*]</button>'
        html += '<br/>'
        html += f'<div class="content">'
        for i, child in enumerate(o.children):
            last = last + [i == length-1]
            html += dfs(child, depth+1, last, name + ".")
            last.pop()

        html += "</div>"
        if ith == 0 or (ith > 1 and ith == len(module_names)-1):
            html += "</div>"
    return html

html = dfs(tree, 0, [], "")

class hello:
    def GET(self, name):
        return render.index(content=html)
class submit:
    def GET(self, _):
        names = web.input().name.split(";")
        print(names)

if __name__ == "__main__":
    app.run()

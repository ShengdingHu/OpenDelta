from transformers import T5Model
from visualization import Visualization
import web

model = T5Model.from_pretrained("t5-small")
vis = Visualization(model)
tree = vis.structure_graph()

urls = (
    '/submit/(.*)', 'submit',
    '/(.*)', 'hello',
)
app = web.application(urls, globals())
render = web.template.render('templates/')

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

def dfs(o, depth, last, name):
    html = ""
    name += o.module_name
    for i in range(depth-1):
        html += prefix0 if last[i] else prefix1
    if depth > 0: html += prefix3 if last[-1] else prefix2
    length = len(o.children)
    if length > 0:
        html += f'<button class="collapsible button_inline">[+]</button>'
    html += f'<button class="selectable button_inline" name="{name}">{colorfy(o.label)}</button><br/>'
    html += f'<div class="content">'
    for i, child in enumerate(o.children):
        last = last + [i == length-1]
        html += dfs(child, depth+1, last, name + ".")
        last.pop()

    html += "</div>"
    return html

html = dfs(tree, 0, [], "")

class hello:
    def GET(self, name):
        return render.test(content=html)

class submit:
    def expand(self, name):
        return [name] # TODO implement in visualization?
    def GET(self, _):
        get_names = web.input().name.split(";")
        names = []
        for name in get_names:
            names += self.expand(name)
        print(names) # TODO

if __name__ == "__main__":
    app.run()

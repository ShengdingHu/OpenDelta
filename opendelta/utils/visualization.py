class Visualization(object):
    def __init__(self, plm):
        self.plm = plm
        pass

    def structure_graph(self):
        print(self.plm)


if __name__=="__main__":
    from openprompt.plms import load_plm
    import argparse
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model", type=str, default='t5-lm', help="We test both t5 and t5-lm in this scripts, the corresponding tokenizerwrapper will be automatically loaded.")
    parser.add_argument("--model_name_or_path", default='/home/hushengding/plm_cache/t5-base-lm-adapt/')
    args = parser.parse_args()
    plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

    visobj = Visualization(plm)
    visobj.structure_graph()


    pass
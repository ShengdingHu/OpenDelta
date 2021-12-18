
t5_mapping = {
    "shared": {"__name__":"embeddings"},
    "lm_head": {"__name__":"lm_head.proj"},
    "encoder": {"__name__":"encoder",
        "embed_tokens": {"__name__":"embeddings"},
        "block": {"__name__":"block",
            "$": {"__name__":"$",
                "layer.0": {"__name__":"attn",
                    "SelfAttention.q": {"__name__":"q"},
                    "SelfAttention.k": {"__name__":"k"},
                    "SelfAttention.v": {"__name__":"v"},
                    "SelfAttention.o": {"__name__":"proj"},
                    "SelfAttention.relative_attention_bias": {"__name__":""},
                    "layer_norm": {"__name__":"layer_norm"},
                },
                "layer.1": {"__name__":"ff",
                    "DenseReluDense.wi_0": {"__name__":""},
                    "DenseReluDense.wi_1": {"__name__":"w1"},
                    "layer_norm": {"__name__":"layer_norm"},
                    "DenseReluDense.wo": {"__name__":"w2"},
                }
            }
        },
        "final_layer_norm": {"__name__":"layer_norm"},  
    },
    "decoder": {"__name__":"decoder",
        "embed_tokens": {"__name__":"embeddings"},
        "block": {"__name__":"block",
            "$": {"__name__":"$",
                "layer.0": {"__name__":"attn",
                    "SelfAttention.q": {"__name__":"q"},
                    "SelfAttention.k": {"__name__":"k"},
                    "SelfAttention.v": {"__name__":"v"},
                    "SelfAttention.o": {"__name__":"proj"},
                    "SelfAttention.relative_attention_bias": {"__name__":""},
                    "layer_norm": {"__name__":"layer_norm"},
                },
                "layer.1": {"__name__":"crossattn",
                    "EncDecAttention.q": {"__name__":"q"},
                    "EncDecAttention.k": {"__name__":"k"},
                    "EncDecAttention.v": {"__name__":"v"},
                    "EncDecAttention.o": {"__name__":"proj"},
                    "layer_norm": {"__name__":"layer_norm"},
                },
                "layer.2": {"__name__":"ff",
                    "DenseReluDense.wi_0": {"__name__":""},
                    "DenseReluDense.wi_1": {"__name__":"w1"},
                    "layer_norm": {"__name__":"layer_norm"},
                    "DenseReluDense.wo": {"__name__":"w2"},
                }
            }
        },
        "final_layer_norm": {"__name__":"layer_norm"},
    }
}


roberta_mapping = {
    "roberta.embeddings.word_embeddings": {"__name__":"embeddings"},
    "roberta.embeddings.position_embeddings": {"__name__":""},
    "roberta.embeddings.token_type_embeddings": {"__name__":""},
    "roberta.embeddings.LayerNorm": {"__name__":""},
    "roberta.encoder": {"__name__":"encoder",
        "layer": {"__name__":"block",
            "$": {"__name__":"$",
                "attention": {"__name__":"attn",
                    "self.query": {"__name__":"q"},
                    "self.key": {"__name__":"k"},
                    "self.value": {"__name__":"v"},
                    "output.dense": {"__name__":"proj"},
                    "output.LayerNorm": {"__name__":"layer_norm"},
                },
                "intermediate.dense": {"__name__":"ff.w1"},
                "output.dense": {"__name__":"ff.w2"},
                "output.LayerNorm": {"__name__":"ff.layer_norm"},
            }
        }
    },
    "lm_head": {"__name__":"lm_head",
        "dense": {"__name__":""},
        "layer_norm": {"__name__":""},
        "decoder": {"__name__":"proj"},
    }
}

bert_mapping = {
    "bert.embeddings.word_embeddings": {"__name__":"embeddings"},
    "bert.embeddings.position_embeddings": {"__name__":""},
    "bert.embeddings.token_type_embeddings": {"__name__":""},
    "bert.embeddings.LayerNorm": {"__name__":""},
    "bert.encoder": {"__name__":"encoder",
        "layer": {"__name__":"block",
            "$": {"__name__":"$",
                "attention": {"__name__":"attn",
                    "self.query": {"__name__":"q"},
                    "self.key": {"__name__":"k"},
                    "self.value": {"__name__":"v"},
                    "output.dense": {"__name__":"proj"},
                    "output.LayerNorm": {"__name__":"layer_norm"},
                },
                "intermediate.dense": {"__name__":"ff.w1"},
                "output.dense": {"__name__":"ff.w2"},
                "output.LayerNorm": {"__name__":"ff.layer_norm"},
            }
        }
    },
    "cls.predictions": {"__name__": "lm_head",
        "transform.dense": {"__name__":""},
        "transform.LayerNorm": {"__name__":""},
        "decoder": {"__name__":"proj"},
    }
}

gpt2_mapping = {
    "transformer.wte": {"__name__":"embeddings"},
    "transformer.wpe": {"__name__":""},
    "transformer.h": {"__name__":"decoder.block",
        "$": {"__name__":"$",
            "attn": {"__name__":"attn",
                "c_attn": {"__name__":"q,k,v"},
                "c_proj": {"__name__":"proj"},
            },
            "ln_1": {"__name__":"attn.layer_norm"},
            "mlp.c_fc": {"__name__":"ff.w1"},
            "mlp.c_proj": {"__name__":"ff.w2"},
            "ln_2": {"__name__":"ff.layer_norm"},
        },
    },
    "transformer.ln_f": {"__name__":"decoder.layernorm"},
    "lm_head": {"__name__":"lm_head.proj"},
}

def transform(org_key, mapping, strict=True, warning=False):
    
    chain = org_key.split(".")
    query = ""
    node = mapping

    new_chain = []
    for elem in chain:
        query += elem
        if query in node:
            node = node[query]
            new_elem = node["__name__"]
            if new_elem == "":
                if strict:
                    if warning:
                        print(f"'{org_key}' has no common mapping.")
                    return 
                else:
                    new_chain.append(query)
            else:
                new_chain.append(new_elem)
            query = ""
        elif "$" in node:
            node = node["$"]
            new_chain.append(query)
            query = ""
        else:
            query += "." 
    if query!="":
        if strict:
            if warning:
                print("A part of the orginial key hasn't been matched!")
            return 
        else:
            new_chain.append(query.strip(".")) # tailing query
    new_key = ".".join(new_chain)
    print(f"{org_key} => {new_key}")
    return new_key
    

Mappings = {
    "t5-lm": t5_mapping,
    "t5": t5_mapping,
    "gpt2": gpt2_mapping,
    "bert": bert_mapping,
    "roberta": roberta_mapping,
}

if __name__ == "__main__":
    from openprompt.plms import load_plm
    import argparse
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model", type=str, default='t5-lm', help="We test both t5 and t5-lm in this scripts, the corresponding tokenizerwrapper will be automatically loaded.")
    parser.add_argument("--model_name_or_path", default="t5-base-lm-adapt")
    parser.add_argument("--cache_base", default='/home/hushengding/plm_cache/')
    parser.add_argument("--keep_non_params", action="store_true")
    parser.add_argument("--expand_params", action="store_true")
    args = parser.parse_args()
    plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.cache_base+args.model_name_or_path)

    for name, _ in plm.named_modules():
        transform(name, t5_mapping, strict=True, warning=False)
    
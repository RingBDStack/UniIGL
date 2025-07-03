from arg import *
import os
import copy
from utils.data_process import *
from utils.metric import acc,mf1,wf1,bacc,auroc
from layers.backbones import *
from model.mixup import mix_feature,mix_emb,Mixer
from layers.loss import *
from model.selection import *
import torch.nn.functional as F
from model.structure import adjust_structure
from layers.mygcn import Prompt_module, GCN
from pretrain import PreTrainer
from finetune import Tuner

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def test(model, prompt, data):
    model.eval()
    prompt.eval()
    with torch.no_grad():
        if args.adj == 'norm':
            node_embeddings = model(data.features, data.adj_norm)
        else:
            node_embeddings = model(data.features, data.adj)
        node_embeddings = prompt(node_embeddings)

        prototypes = prompt.compute_prototype_logits(
            node_embeddings,
            labels=data.labels,
            train_idx=data.train_idx,
            tau2=args.tau2,
            return_prototypes=True
        )

        test_embeddings = node_embeddings[data.test_idx]
        test_sim = F.cosine_similarity(
            test_embeddings.unsqueeze(1),
            prototypes.unsqueeze(0),
            dim=-1
        ) / args.tau2

        all_outputs = torch.softmax(test_sim, dim=-1)
        all_labels  = torch.argmax(data.labels[data.test_idx], dim=-1)

        acc_test   = acc(all_labels, all_outputs)
        mf1_test   = mf1(all_labels, all_outputs)
        wf1_test   = wf1(all_labels, all_outputs)
        bacc_test  = bacc(all_labels, all_outputs)
        auroc_test = auroc(all_labels, all_outputs)

        print(f'ACC TEST: {acc_test:.4f} || bACC TEST: {bacc_test:.4f} '
              f'|| WF1 TEST: {wf1_test:.4f} || MF1 TEST: {mf1_test:.4f} '
              f'|| ROC TEST: {auroc_test:.4f}')

    res_dir = os.path.join(os.getcwd(), "res")
    os.makedirs(res_dir, exist_ok=True)
    file_path = os.path.join(res_dir, f"{args.dataset}.txt")
    with open(file_path, "a") as f:
        f.write(f"{acc_test:.4f}\t{bacc_test:.4f}\t{mf1_test:.4f}\t{wf1_test:.4f}\n")

    return acc_test, bacc_test, mf1_test, wf1_test
                    
if __name__=="__main__":
    # args = get_parser().parse_args()
    # print(args)
    parser = get_parser()
    args = update_params(parser)

    set_seed(args.random_seed)
    data = load_processed_data(args)
    data.labels = F.one_hot(data.labels, data.nclass).float()
    args.mixnum = (int)(args.mixnum * data.num_nodes * 0.1)

    data.adj_norm = normalize_adj_torch(data.adj)
    data = data.to('cuda')
    data.train_mask = data.train_mask.to('cuda')
    
# --------------------------------Pre-Train--------------------------------
    save_dir = os.path.join(os.getcwd(), "saved")
    save_path = f'{args.dataset}_{args.imbtype}_preGNN.pth'
    struc_path = f'{args.dataset}_{args.imbtype}_structure.pth'
    
    model_path = os.path.join(save_dir, save_path)
    structure_path = os.path.join(save_dir, struc_path)
    
    os.system(f'rm -f {model_path}')
    
    trainer = PreTrainer(args, data, model_path, structure_path)
    _, data.adj, data.adj_nrom = trainer.train()
    
    fineGNN = GCN(data.features.shape[1], args.nhid, data.labels.shape[1], 
                         args.nlayer, args.dropout)
    fineGNN.load_pretrained_weights(model_path)
    fineGNN = fineGNN.to('cuda')
    
# --------------------------------Prompting--------------------------------
    with torch.no_grad():
        fineGNN.eval()
        emb = fineGNN(data.features, data.adj_norm)
        num_classes = data.labels.shape[1]
        prototypes = torch.zeros(num_classes, emb.size(1), device=emb.device)
        # emb = F.normalize(emb, dim=-1)
        for c in range(num_classes):
            class_weights = data.labels[data.train_idx, c]
            weighted_sum = torch.sum(emb[data.train_idx] * class_weights.unsqueeze(-1), dim=0)
            total_weight = class_weights.sum()
            
            prototypes[c] = weighted_sum / total_weight    
    
    # prototypes = F.normalize(prototypes, dim=-1)    
        
    logits = F.cosine_similarity(
        emb.unsqueeze(1),
        prototypes.unsqueeze(0),
        dim=-1
    ) / args.tau2
            

    logits_mc = fineGNN.compute_second_term_logits(
        data.features, data.adj_norm, prototypes, args.tau2)            
    
    mix_pairs = information_select(
        logits_det=logits,
        logits_mc=logits_mc,
        lambda_value=args.lam,
        alpha=args.alpha,
        num=args.mixnum,
        train_idx=data.train_idx,
        labeled_nodes=data.labeled_nodes,
    )

    new_features, new_labels, new_train_idx, new_adj = mix_feature(data, mix_pairs, args.lam, args)
    new_adj_norm = normalize_adj_torch(new_adj)
    
    new_data = New_Data(new_adj, new_adj_norm, new_features, new_labels, 
                        new_train_idx, data.val_idx, data.test_idx)
    new_data = new_data.to('cuda')
    
    tuner = Tuner(args, new_data, model_path, data)
    best_model, best_prompt = tuner.train()
            
    test(model=best_model, prompt=best_prompt, data=data)
    
    os.system(f'rm -f {model_path}')


    

import torch
import os
from tqdm import tqdm
import numpy as np
import argparse

# Import saliency methods and models
from ViT_explanation_generator import Baselines
from ViT_new import vit_base_patch16_224
# from models.vgg import vgg19
import glob

from dataset.expl_hdf5 import ImagenetResults


def normalize(tensor,
              mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


def eval(args):
    num_samples = 0
    num_correct_model = np.zeros((len(imagenet_ds,)))
    dissimilarity_model = np.zeros((len(imagenet_ds,)))
    model_index = 0

    if args.scale == 'per':
        base_size = 224 * 224
        perturbation_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    elif args.scale == '100':
        base_size = 100
        perturbation_steps = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    else:
        raise Exception('scale not valid')

    num_correct_pertub = np.zeros((9, len(imagenet_ds)))
    dissimilarity_pertub = np.zeros((9, len(imagenet_ds)))
    logit_diff_pertub = np.zeros((9, len(imagenet_ds)))
    prob_diff_pertub = np.zeros((9, len(imagenet_ds)))
    perturb_index = 0

    for batch_idx, (data, vis, target) in enumerate(tqdm(sample_loader)):
        # Update the number of samples
        num_samples += len(data)

        data = data.to(device)
        vis = vis.to(device)
        target = target.to(device)
        norm_data = normalize(data.clone())

        # Compute model accuracy
        pred = model(norm_data)
        pred_probabilities = torch.softmax(pred, dim=1)
        pred_org_logit = pred.data.max(1, keepdim=True)[0].squeeze(1)
        pred_org_prob = pred_probabilities.data.max(1, keepdim=True)[0].squeeze(1)
        pred_class = pred.data.max(1, keepdim=True)[1].squeeze(1)
        tgt_pred = (target == pred_class).type(target.type()).data.cpu().numpy()
        num_correct_model[model_index:model_index+len(tgt_pred)] = tgt_pred

        probs = torch.softmax(pred, dim=1)
        target_probs = torch.gather(probs, 1, target[:, None])[:, 0]
        second_probs = probs.data.topk(2, dim=1)[0][:, 1]
        temp = torch.log(target_probs / second_probs).data.cpu().numpy()
        dissimilarity_model[model_index:model_index+len(temp)] = temp

        if args.wrong:
            wid = np.argwhere(tgt_pred == 0).flatten()
            if len(wid) == 0:
                continue
            wid = torch.from_numpy(wid).to(vis.device)
            vis = vis.index_select(0, wid)
            data = data.index_select(0, wid)
            target = target.index_select(0, wid)

        # Save original shape
        org_shape = data.shape

        if args.neg:
            vis = -vis

        vis = vis.reshape(org_shape[0], -1)

        for i in range(len(perturbation_steps)):
            _data = data.clone()

            _, idx = torch.topk(vis, int(base_size * perturbation_steps[i]), dim=-1)
            idx = idx.unsqueeze(1).repeat(1, org_shape[1], 1)
            _data = _data.reshape(org_shape[0], org_shape[1], -1)
            _data = _data.scatter_(-1, idx, 0)
            _data = _data.reshape(*org_shape)

            _norm_data = normalize(_data)

            out = model(_norm_data)

            pred_probabilities = torch.softmax(out, dim=1)
            pred_prob = pred_probabilities.data.max(1, keepdim=True)[0].squeeze(1)
            diff = (pred_prob - pred_org_prob).data.cpu().numpy()
            prob_diff_pertub[i, perturb_index:perturb_index+len(diff)] = diff

            pred_logit = out.data.max(1, keepdim=True)[0].squeeze(1)
            diff = (pred_logit - pred_org_logit).data.cpu().numpy()
            logit_diff_pertub[i, perturb_index:perturb_index+len(diff)] = diff

            target_class = out.data.max(1, keepdim=True)[1].squeeze(1)
            temp = (target == target_class).type(target.type()).data.cpu().numpy()
            num_correct_pertub[i, perturb_index:perturb_index+len(temp)] = temp

            probs_pertub = torch.softmax(out, dim=1)
            target_probs = torch.gather(probs_pertub, 1, target[:, None])[:, 0]
            second_probs = probs_pertub.data.topk(2, dim=1)[0][:, 1]
            temp = torch.log(target_probs / second_probs).data.cpu().numpy()
            dissimilarity_pertub[i, perturb_index:perturb_index+len(temp)] = temp

        model_index += len(target)
        perturb_index += len(target)

    np.save(os.path.join(args.experiment_dir, 'model_hits.npy'), num_correct_model)
    np.save(os.path.join(args.experiment_dir, 'model_dissimilarities.npy'), dissimilarity_model)
    np.save(os.path.join(args.experiment_dir, 'perturbations_hits.npy'), num_correct_pertub[:, :perturb_index])
    np.save(os.path.join(args.experiment_dir, 'perturbations_dissimilarities.npy'), dissimilarity_pertub[:, :perturb_index])
    np.save(os.path.join(args.experiment_dir, 'perturbations_logit_diff.npy'), logit_diff_pertub[:, :perturb_index])
    np.save(os.path.join(args.experiment_dir, 'perturbations_prob_diff.npy'), prob_diff_pertub[:, :perturb_index])

    print(np.mean(num_correct_model), np.std(num_correct_model))
    print(np.mean(dissimilarity_model), np.std(dissimilarity_model))
    print(perturbation_steps)
    print(np.mean(num_correct_pertub, axis=1), np.std(num_correct_pertub, axis=1))
    print(np.mean(dissimilarity_pertub, axis=1), np.std(dissimilarity_pertub, axis=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a segmentation')
    parser.add_argument('--batch-size', type=int,
                        default=16,
                        help='')
    parser.add_argument('--neg', type=bool,
                        default=True,
                        help='')
    parser.add_argument('--value', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--scale', type=str,
                        default='per',
                        choices=['per', '100'],
                        help='')
    parser.add_argument('--method', type=str,
                        default='grad_rollout',
                        choices=['rollout', 'lrp', 'transformer_attribution', 'full_lrp', 'v_gradcam', 'lrp_last_layer',
                                 'lrp_second_layer', 'gradcam',
                                 'attn_last_layer', 'attn_gradcam', 'input_grads'],
                        help='')
    parser.add_argument('--vis-class', type=str,
                        default='top',
                        choices=['top', 'target', 'index'],
                        help='')
    parser.add_argument('--wrong', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--class-id', type=int,
                        default=0,
                        help='')
    parser.add_argument('--is-ablation', type=bool,
                        default=False,
                        help='')
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    # PATH variables
    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    dataset = PATH + 'dataset/'
    os.makedirs(os.path.join(PATH, 'experiments'), exist_ok=True)
    os.makedirs(os.path.join(PATH, 'experiments/perturbations'), exist_ok=True)

    exp_name = args.method
    exp_name += '_neg' if args.neg else '_pos'
    print(exp_name)

    if args.vis_class == 'index':
        args.runs_dir = os.path.join(PATH, 'experiments/perturbations/{}/{}_{}'.format(exp_name,
                                                                                       args.vis_class,
                                                                                       args.class_id))
    else:
        ablation_fold = 'ablation' if args.is_ablation else 'not_ablation'
        args.runs_dir = os.path.join(PATH, 'experiments/perturbations/{}/{}/{}'.format(exp_name,
                                                                                    args.vis_class, ablation_fold))
        # args.runs_dir = os.path.join(PATH, 'experiments/perturbations/{}/{}'.format(exp_name,
        #                                                                             args.vis_class))

    if args.wrong:
        args.runs_dir += '_wrong'

    experiments = sorted(glob.glob(os.path.join(args.runs_dir, 'experiment_*')))
    experiment_id = int(experiments[-1].split('_')[-1]) + 1 if experiments else 0
    args.experiment_dir = os.path.join(args.runs_dir, 'experiment_{}'.format(str(experiment_id)))
    os.makedirs(args.experiment_dir, exist_ok=True)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    if args.vis_class == 'index':
        vis_method_dir = os.path.join(PATH,'visualizations/{}/{}_{}'.format(args.method,
                                                          args.vis_class,
                                                          args.class_id))
    else:
        ablation_fold = 'ablation' if args.is_ablation else 'not_ablation'
        vis_method_dir = os.path.join(PATH,'visualizations/{}/{}/{}'.format(args.method,
                                                       args.vis_class, ablation_fold))
        # vis_method_dir = os.path.join(PATH, 'visualizations/{}/{}'.format(args.method,
        #                                                                      args.vis_class))

    # imagenet_ds = ImagenetResults('visualizations/{}'.format(args.method))
    imagenet_ds = ImagenetResults(vis_method_dir)

    # Model
    model = vit_base_patch16_224(pretrained=True).cuda()
    model.eval()

    save_path = PATH + 'results/'

    sample_loader = torch.utils.data.DataLoader(
        imagenet_ds,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=False)

    eval(args)

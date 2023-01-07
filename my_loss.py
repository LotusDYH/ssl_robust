import torch
import diffdist.functional as distops


def get_loss(similarity_matrix, device):

    size = len(similarity_matrix)
    batch_size = int(len(similarity_matrix) / 2)

    similarity_matrix_exp = torch.exp(similarity_matrix)
    similarity_matrix_exp = similarity_matrix_exp * (1 - torch.eye(size, size)).to(device)

    loss = -torch.log(
        similarity_matrix_exp / (torch.sum(similarity_matrix_exp, dim=1).view(size, 1) + 1e-8) + 1e-8)

    loss_total = (1. / float(size)) * torch.sum(
        torch.diag(loss[0: batch_size, batch_size:]) + torch.diag(loss[batch_size:, 0: batch_size]))
    return loss_total


def pairwise_similarity(outputs, ngpu, temperature=0.5):

    if ngpu > 1:
        batch_size = int(len(outputs) / 2)
        outputs_1 = outputs[0:batch_size]
        outputs_2 = outputs[batch_size:]

        gather_1 = [torch.empty_like(outputs_1) for _ in range(ngpu)]
        gather_1 = distops.all_gather(gather_1, outputs_1)
        gather_2 = [torch.empty_like(outputs_2) for _ in range(ngpu)]
        gather_2 = distops.all_gather(gather_2, outputs_2)

        outputs_1 = torch.cat(gather_1)
        outputs_2 = torch.cat(gather_2)
        outputs = torch.cat((outputs_1, outputs_2))

    size = len(outputs)
    outputs_norm = outputs / (outputs.norm(dim=1).view(size, 1) + 1e-8)
    similarity_matrix = (1. / temperature) * torch.mm(outputs_norm, outputs_norm.transpose(0, 1).detach())

    return similarity_matrix

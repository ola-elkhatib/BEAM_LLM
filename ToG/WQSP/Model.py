
import torch
from torch import nn, Tensor



class Model(nn.Module):

    def __init__(self, pretrained_embeddings,dropout_val,do_batchnorm,do_dropout,freeze=False):
        super().__init__()
        self.encoder_head = nn.Embedding.from_pretrained(torch.stack(pretrained_embeddings[0]), freeze=freeze)
        self.encoder_rel = nn.Embedding.from_pretrained(torch.stack(pretrained_embeddings[1]), freeze=freeze)
        self.dropout = torch.nn.Dropout(dropout_val)
        self.batchnorm1 = nn.BatchNorm1d(2048)

        self.dropout_rel = torch.nn.Dropout(dropout_val)
        self.do_batchnorm = do_batchnorm
        self.do_dropout = do_dropout

        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn2 = torch.nn.BatchNorm1d(2)

    def forward(self, head, relation) -> Tensor:
        head = self.encoder_head(head)
        relations = self.encoder_rel(relation)
        hops = relations.shape[1]
        if hops == 1: R = relations.squeeze(1)
        if hops == 2:
            R = self.new_operation(relations[:, 0, :], relations[:, 1, :])
        if hops == 3:
            R = self.new_operation(relations[:, 0, :], relations[:, 1, :])
            R = self.new_operation(R, relations[:, 2, :])

        pred = self.ComplEx(head, R)
        for r in range(relations.shape[1]):
            rp1 = relations[:, r, :]
            pred2 = self.ComplEx(head, rp1)
            new_head = pred2.argmax(dim=1)
            head = self.encoder_head(new_head)
        return pred, pred2

    def another_forward(self, head, relation) -> Tensor:
        head = self.encoder_head(head)
        relations = self.encoder_rel(relation)
        pred = self.ComplEx(head, relations)
        return pred

    def extract(self, emb):
        real, imaginary = torch.chunk(emb, 2, dim=1)
        return real, imaginary

    def new_operation(self, r1, r2):
        re_r1, im_r1 = self.extract(r1)
        re_r2, im_r2 = self.extract(r2)
        re_r = re_r1 * re_r2 - im_r1 * im_r2
        im_r = re_r1 * im_r2 + im_r1 * re_r2
        r = torch.cat([re_r, im_r], dim=1)
        return r

    def get_score_ranked(self, preds):
        top2 = torch.topk(preds, k=2, largest=True, sorted=True)
        return top2, None

    # inspired by https://github.com/malllabiisc/EmbedKGQA
    def ComplEx(self, head, relation):
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        if self.do_batchnorm :
            head = self.bn0(head)
        if self.do_dropout :
            head = self.dropout(head)
            relation = self.dropout(relation)
        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]

        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.encoder_head.weight, 2, dim=1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = torch.stack([re_score, im_score], dim=1)
        if self.do_batchnorm :
            score = self.bn2(score)
        if self.do_dropout : score = self.dropout(score)
        score = score.permute(1, 0, 2)

        re_score = score[0]
        im_score = score[1]
        score = torch.mm(re_score, re_tail.transpose(1, 0)) + torch.mm(im_score, im_tail.transpose(1, 0))
        return score

    def score_relations_with_tails(self, head):
        """
        Scores every relation with respect to every tail entity and aggregates the scores.
        """
        # Get the head embedding
        head = self.encoder_head(head)
        relations = self.encoder_rel.weight  # Shape: (num_relations, embedding_dim)
        num_relations = relations.shape[0]

        # Split the head embedding into real and imaginary parts
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        if self.do_batchnorm:
            head = self.bn0(head)
        if self.do_dropout:
            head = self.dropout(head)

        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]

        # Split the relation embeddings into real and imaginary parts
        re_relation, im_relation = torch.chunk(relations, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.encoder_head.weight, 2, dim=1)

        # Compute scores for every relation and every tail entity
        re_score = re_head.unsqueeze(1) * re_relation - im_head.unsqueeze(1) * im_relation
        im_score = re_head.unsqueeze(1) * im_relation + im_head.unsqueeze(1) * re_relation

        # Aggregate scores for each relation across all tail entities
        re_score = torch.matmul(re_score, re_tail.transpose(1, 0))
        im_score = torch.matmul(im_score, im_tail.transpose(1, 0))
        score = re_score + im_score  # Shape: (batch_size, num_relations, num_tails)

        # Sum scores over all tail entities
        relation_scores = score.sum(dim=2)  # Shape: (batch_size, num_relations)

        return relation_scores
    

        
    def score_relations_with_tails_libkge(self, head):
        """
        Scores every relation with respect to every tail entity and aggregates the scores for each relation.
        """
        # Get the relation and tail embeddings
        s_emb = self.encoder_head(head)
        p_emb = self.encoder_rel.weight  # All relation embeddings
        o_emb = self.encoder_head.weight  # All tail entity embeddings

        n = p_emb.size(0)  # Number of relations

        # Split the embeddings into real and imaginary parts
        p_emb_re, p_emb_im = (t.contiguous() for t in p_emb.chunk(2, dim=1))
        o_emb_re, o_emb_im = (t.contiguous() for t in o_emb.chunk(2, dim=1))

        # Combine subject (head), relation, and object (tail) embeddings
        s_all = torch.cat((s_emb, s_emb), dim=1)  # re, im, re, im
        r_all = torch.cat((p_emb_re, p_emb_re, p_emb_im, -p_emb_im), dim=1)  # re, re, im, -im
        o_all = torch.cat((o_emb_re, o_emb_im, o_emb_im, o_emb_re), dim=1)  # re, im, im, re

        # Compute the scores for all relations with all tails
        scores = (s_all.unsqueeze(0) * r_all.unsqueeze(1) * o_all.unsqueeze(0)).sum(dim=2)

        # Aggregate the scores for each relation across all tails
        relation_scores = scores.sum(dim=1)  # Shape: (num_relations,)

        return relation_scores
    

    def ComplEx_relation(self, head):
        # Embed the head entity
        head = self.encoder_head(head)  # Shape: [batch_size, embedding_dim]
        
        # Split head embeddings into real and imaginary parts
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)  # Shape: [2, batch_size, embedding_dim / 2]
        if self.do_batchnorm:
            head = self.bn0(head)
        head = self.dropout(head)

        # Permute to separate real and imaginary parts
        head = head.permute(1, 0, 2)  # Shape: [2, batch_size, embedding_dim / 2]
        re_head, im_head = head[0], head[1]  # Real and imaginary parts of head

        # Get all tail embeddings and split into real and imaginary parts
        re_tail, im_tail = torch.chunk(self.encoder_head.weight, 2, dim=1)  # Shape: [num_tails, embedding_dim / 2]

        # Get all relation embeddings and split into real and imaginary parts
        re_relation, im_relation = torch.chunk(self.encoder_rel.weight, 2, dim=1)  # Shape: [num_relations, embedding_dim / 2]

        # Compute scores for each relation for all tails
        scores = []
        for i in range(re_relation.size(0)):  # Iterate over all relations
            # Relation real and imaginary parts
            re_r = re_relation[i].unsqueeze(0)  # Shape: [1, embedding_dim / 2]
            im_r = im_relation[i].unsqueeze(0)  # Shape: [1, embedding_dim / 2]

            # Compute ComplEx score for all tails
            re_score = torch.mm(re_head * re_r - im_head * im_r, re_tail.T)  # Real part
            im_score = torch.mm(re_head * im_r + im_head * re_r, im_tail.T)  # Imaginary part

            # Combine scores
            relation_score = re_score + im_score  # Shape: [batch_size, num_tails]

            # Aggregate over all tails (e.g., mean aggregation)
            aggregated_score = relation_score.mean(dim=1)  # Shape: [batch_size]
            scores.append(aggregated_score)

        # Stack scores for all relations
        scores = torch.stack(scores, dim=1)  # Shape: [batch_size, num_relations]

        # Apply sigmoid to normalize scores
        pred = torch.sigmoid(scores)  # Shape: [batch_size, num_relations]

        return pred

    def ComplEx_relation_tail(self, head, tail):
        # Embed the head and tail entities
        head = self.encoder_head(head)  # Shape: [batch_size, embedding_dim]
        tail = self.encoder_head(tail)  # Shape: [batch_size, embedding_dim]

        # Split the head and tail embeddings into real and imaginary parts
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)  # Shape: [2, batch_size, embedding_dim / 2]
        tail = torch.stack(list(torch.chunk(tail, 2, dim=1)), dim=1)  # Shape: [2, batch_size, embedding_dim / 2]

        if self.do_batchnorm:
            head = self.bn0(head)
            tail = self.bn0(tail)

        head = self.dropout(head)
        tail = self.dropout(tail)

        # Separate real and imaginary parts of the head and tail
        head = head.permute(1, 0, 2)  # Shape: [2, batch_size, embedding_dim / 2]
        tail = tail.permute(1, 0, 2)  # Shape: [2, batch_size, embedding_dim / 2]
        re_head, im_head = head[0], head[1]
        re_tail, im_tail = tail[0], tail[1]

        # Get all relation embeddings and split into real and imaginary parts
        re_relation, im_relation = torch.chunk(self.encoder_rel.weight, 2, dim=1)  # Shape: [num_relations, embedding_dim / 2]

        # Compute scores for all relations
        re_score = re_head * re_relation + im_head * im_relation  # Real part
        im_score = im_head * re_relation - re_head * im_relation  # Imaginary part
       
        # Combine real and imaginary parts and compute the dot product with tail
        score = torch.mm(re_score, re_tail.T) + torch.mm(im_score, im_tail.T)  # Shape: [batch_size, num_relations]
        breakpoint()
        # Apply sigmoid to normalize scores
        pred = torch.sigmoid(score)  # Shape: [batch_size, num_relations]

        return pred
"""One balanced draw sequence, disjoint draw positions, no evaluation padding."""
from collections import Counter
import hashlib
import torch
from torch.utils.data import Sampler


class GlobalBalancedSampler(Sampler):
    def __init__(self,labels,rank,world_size,seed=42,epoch=0,batch_offset=0,batch_size=2):
        self.labels=list(labels);self.rank=rank;self.world_size=world_size
        if set(labels)!={0,1} or len(labels)%world_size:raise ValueError('Two classes and divisible epoch size required')
        if not 0<=rank<world_size:raise ValueError('Invalid rank')
        self.seed=seed;self.batch_size=batch_size;self.set_epoch(epoch,batch_offset)

    def set_epoch(self,epoch,batch_offset=0):
        self.epoch=epoch;self.batch_offset=batch_offset
        self.generator=torch.Generator().manual_seed(self.seed+epoch)
        pieces=[]
        for label in (0,1):
            pool=torch.tensor([i for i,y in enumerate(self.labels) if y==label])
            n=len(self.labels)//2+(len(self.labels)%2 if label==1 else 0)
            pieces.append(pool[torch.randint(len(pool),(n,),generator=self.generator)])
        sequence=torch.cat(pieces)
        self.sequence=sequence[torch.randperm(len(sequence),generator=self.generator)].tolist()
        self.local=self.sequence[self.rank::self.world_size]
        if batch_offset<0 or batch_offset*self.batch_size>len(self.local):raise ValueError('Invalid offset')

    def __iter__(self):return iter(self.local[self.batch_offset*self.batch_size:])
    def __len__(self):return len(self.local)-self.batch_offset*self.batch_size
    def state_dict(self):
        return dict(epoch=self.epoch,batch_offset=self.batch_offset,seed=self.seed,rank=self.rank,
                    world_size=self.world_size,batch_size=self.batch_size,generator_state=self.generator.get_state(),
                    sequence_sha256=hashlib.sha256(str(self.sequence).encode()).hexdigest())
    def load_state_dict(self,state):
        for key in ('seed','rank','world_size','batch_size'):
            if state[key]!=getattr(self,key):raise ValueError('Sampler resume mismatch: '+key)
        self.set_epoch(state['epoch'],state['batch_offset'])
        if state['sequence_sha256']!=self.state_dict()['sequence_sha256']:raise ValueError('Sampler sequence mismatch')
        self.generator.set_state(state['generator_state'])
    def summary(self):
        counts=Counter(self.sequence)
        shards=[self.sequence[r::self.world_size] for r in range(self.world_size)]
        shard_sets=list(map(set,shards))
        return dict(global_samples=len(self.sequence),rank_samples=list(map(len,shards)),
            labels=dict(Counter(self.labels[i] for i in self.sequence)),unique_samples=len(counts),
            repeated_draws=len(self.sequence)-len(counts),max_multiplicity=max(counts.values()),
            cross_rank_shared_sample_ids=sum(sum(i in s for s in shard_sets)>1 for i in counts),
            draw_position_overlap=0,definition='Shards partition draw positions; replacement permits repeated sample IDs, including across ranks')


class UnpaddedDistributedSampler(Sampler):
    def __init__(self,size,rank,world_size):self.indices=list(range(rank,size,world_size))
    def __iter__(self):return iter(self.indices)
    def __len__(self):return len(self.indices)

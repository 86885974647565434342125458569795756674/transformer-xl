# transformer-xl

[kimiyoung/transformer-xl (github.com)](https://github.com/kimiyoung/transformer-xl/tree/master)



```
docker run --privileged -itd --ipc host --network host --gpus all --name -jqxx -v /data//jiqixuexi/transformer-xl:/share  nvcr.io/nvidia/pytorch:23.10-py3
docker exec --privileged -it -jqxx /bin/bash
ctrl+D
passwd
apt update
apt install -y openssh-server
Port 17000
service ssh start
docker restart -jqxx
:266
?向上搜索
/向下搜索
复制粘贴：vyp
:Vexplore #竖直分割窗口打开，选中文件就可以打开
:vs 文件名:分割当前窗口
ctrl w w/h/j/k/l:切换窗口
gg:开头
G:结尾
n:往下
N:往上
```

# 源码

```
Namespace(data='../data/enwik8/', dataset='enwik8', n_layer=6, n_head=4, d_head=32, d_embed=-1, d_model=128, d_inner=1024, dropout=0.1, dropatt=0.0, init='normal', emb_init='normal', init_range=0.1, emb_init_range=0.01, init_std=0.02, proj_init_std=0.01, optim='adam', lr=0.00012, mom=0.0, scheduler='cosine', warmup_step=0, decay_rate=0.5, lr_min=0.0, clip=0.25, clip_nonemb=False, max_step=1000, batch_size=11, batch_chunk=1, tgt_len=256, eval_tgt_len=64, ext_len=0, mem_len=256, not_tied=False, seed=1111, cuda=True, adaptive=False, div_val=1, pre_lnorm=False, varlen=False, multi_gpu=False, log_interval=200, eval_interval=100, work_dir='enwik8_small', restart=False, restart_dir='', debug=False, same_length=False, attn_type=0, clamp_len=-1, eta_min=0.0, gpu0_bsz=2, max_eval_steps=-1, sample_softmax=-1, patience=0, finetune_v2=False, finetune_v3=False, fp16=False, static_loss_scale=1, dynamic_loss_scale=False, tied=True)
```

n_token=204

d_embed = d_model=128

256+0+256

self.max_klen = tgt_len + ext_len + mem_len

n_layer=6

```
self.word_emb = AdaptiveEmbedding
	nn.Embedding(n_token, d_embed, sparse=sample_softmax>0)
RelPartialLearnableDecoderLayer
	self.dec_attn = RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn)
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)
	self.pos_ff = PositionwiseFF
        self.CoreNet = nn.Sequential(
        nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(d_inner, d_model),
        nn.Dropout(dropout),
        )
self.crit = ProjectedAdaptiveLogSoftmax
self.pos_emb = PositionalEmbedding(self.d_model)
	inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))#所有layer共享
self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
    device=device, ext_len=args.ext_len)
```

```
for batch, (data, target, seq_len) in enumerate(train_iter):
#torch.Size([256, 11]),torch.Size([256, 11]),256
    ret = para_model(data, target, *mems)
    loss, mems = ret[0], ret[1:]
```

MemTransformerLM.forward(self, data, target, *mems):

```
mems=[...]#n_layer+1，上一次模型前向的中间值
hidden, new_mems = self._forward(data, mems=mems)#最后一层输出，
pred_hid = hidden[-tgt_len:]#label的长度，应该就是每次输入的长度
loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
loss = loss.view(tgt_len, -1)
return [loss] + new_mems
```

MemTransformerLM._forward(self, dec_inp, mems=None):

```
word_emb = self.word_emb(dec_inp)
#torch.Size([256, 11])->torch.Size([256, 11, 128])
klen = mlen + qlen#(0\256)+256
dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1+mlen).byte()[:,:,None]
#[qlen,klen,1]，0是要关注的，[i,j]=0,j<=i+mlen
pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device, 
                                   dtype=word_emb.dtype)
#[255/(255+256),...,0],到自己的距离
pos_emb = self.pos_emb(pos_seq)
#torch.Size([256/(256+256)])->torch.Size([256/(256+256), 1, 128])
#256/(256+256)个位置，每个位置由一向量表示
#pos_emb[-1]=[0,..,1,...]
core_out = self.drop(word_emb)
hids.append(core_out)#第i层的输入，i-1层的输出，这次模型前向的中间值
for i, layer in enumerate(self.layers):
    mems_i = None if mems is None else mems[i]
    core_out = layer(core_out, pos_emb, self.r_w_bias,
    self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
    #[256, 11, 128],[255/(255+256), 1, 128],[n_head, d_head],[256,256/(256+256),1]
    hids.append(core_out)
new_mems = self._update_mems(hids, mems, mlen, qlen)     
#新中间结果，旧中间结果，这里有bug，函数原型和函数定义不一样，但刚好数据一样
```

RelPartialLearnableDecoderLayer.forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

RelPartialLearnableMultiHeadAttn.forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):

```
qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)
#256,256/(256+256),11
cat = torch.cat([mems, w], 0)#[(0/256)+256, 11, 128]=X
w_heads = self.qkv_net(cat)#[(0/256)+256, 11, 3 * n_head * d_head]
#XW_QKV=QKV
r_head_k = self.r_net(r)#[(0/256)+256, 1, n_head * d_head]
#不同位置，不同表示
w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)#[(0/256)+256, 11, n_head * d_head]
w_head_q = w_head_q[-qlen:]#[256, 11, n_head * d_head],Q不需要前面的，KV要
rw_head_q = w_head_q + r_w_bias
#Q+xx
AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))
#[256, 11, n_head, d_head],[(0/256)+256, 11, n_head, d_head]->[256,(0/256)+256,11,n_head]
#QK
rr_head_q = w_head_q + r_r_bias
#Q+xx
BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))   
#[256, 11, n_head, d_head],[(0/256)+256, n_head, d_head]->[256,(0/256)+256,11,n_head]
#QK
BD = self._rel_shift(BD)#?
#[256,(0/256)+256,...]
#[256,1+(0/256)+256,...]
#[1+(0/256)+256,256,...]
#[256,(0/256)+256,...]
#BD的每一行都是对不同位置的信息，要末尾的信息，最后一个是自己的位置，整个矩阵要去掉上三角，每一行有效信息往前移
#相当于从下往上挤
#每行前面加上0，挤掉第一行
#[i,j]有效，j<=i+(0/256)

attn_score = AC + BD
attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -float('inf')).type_as(attn_score)
#attn_score[i][j]=-inf,j>i+(0/256)
attn_prob = F.softmax(attn_score, dim=1)
#高维独立，1维元素之间，对应元素softmax
#相同(256,x,11,4)下标的元素softmax
#[256,256+(0/256),11,4]
attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))
#[256,256+(0/256),11,n_head],[256+(0/256), 11, n_head, d_head]->[256,11,n_head,d_head]
...
[qlen,bsz,d_model]
```

_update_mems:实际就是把hids->new_mems

```
        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context.(会再次加入qlen，不要管ext，不会用到) Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.（每次只保存过去mem_len长度的句子）
			new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())
```

evaluate:

eval_batch_size = 10

va_iter = corpus.get_iterator('valid', eval_batch_size, args.eval_tgt_len,
    device=device, ext_len=args.ext_len)
te_iter = corpus.get_iterator('test', eval_batch_size, args.eval_tgt_len,
    device=device, ext_len=args.ext_len)

```
model.reset_length(args.eval_tgt_len,
args.ext_len, args.mem_len+args.tgt_len-args.eval_tgt_len)
#tgt_len, ext_len, mem_len=64,0,256+256-64
ret = model(data, target, *mems)
#[64,10]
```



```
Namespace(data='../data/enwik8/', dataset='enwik8', split='test', batch_size=10, tgt_len=80, ext_len=0, mem_len=2100, clamp_len=820, cuda=True, work_dir='enwik8_small-enwik8/20231214-013645', no_log=False, same_length=True)
```

```
if args.clamp_len > 0:
    model.clamp_len = args.clamp_len
if args.same_length:
    model.same_length = True
```

_forward:

```
klen = mlen + qlen
#过去+现在
if self.same_length:
	all_ones = word_emb.new_ones(qlen, klen)
    mask_len = klen - self.mem_len
    #超过mem的长度不要关注，训练时是关注过去+现在，推测时是关注过去+现在最新的mem，关注长度相同了
    #只关注最多self.mem_len
    #mask每一行的最后一个元素才是表示最新token
    if mask_len > 0:
    	mask_shift_len = qlen - mask_len
    else:
    	mask_shift_len = qlen
    dec_attn_mask = (torch.triu(all_ones, 1+mlen)
    + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None] # -1
    #0是要关注的，triu0往上挤，tril0往下挤
if self.clamp_len > 0:
    pos_seq.clamp_(max=self.clamp_len)
    #最远的相对位置，没有改变atten数量，超过这个距离都是这个距离
```

_update_mems原型错误

最多保留self.mem_len

没有实现加速的实验

2 for Vaswani et al，mem_len=0，普通

3 for Al Rfou et al，切段，应该也是没有mem_len

```
self.dec_attn = MultiHeadAttn
elif self.attn_type == 2: # absolute standard
	self.pos_emb = PositionalEmbedding(self.d_model)
elif self.attn_type == 3: # absolute deeper SA
	self.r_emb = nn.Parameter(torch.Tensor(
		self.n_layer, self.max_klen, self.n_head, self.d_head))
```

扩大mem

u，v改成不同层不一样

attention范围一样

最高层attention n_layer+1个段，最低层attention2个段，不平衡，mem低层扩大

```
bash run_enwik8_small_vanilla.sh
bash run_enwik8_small_xl.sh
bash run_enwik8_small_.sh
```

# vanilla

`attn_type=2` and `mem_len=0`

self.word_emb = AdaptiveEmbedding

​	nn.Embedding

DecoderLayer:

​	MultiHeadAttn:

​		self.q_net = nn.Linear

​		self.kv_net = nn.Linear

​		self.o_net = nn.Linear

​	PositionwiseFF:

self.pos_emb = PositionalEmbedding(self.d_model)

MemTransformerLM.forward:

```
if not mems: mems = self.init_mems()
#mems=None
```

_forward:

```
mlen = mems[0].size(0) if mems is not None else 0
#mlen=0
klen = mlen + qlen
core_out = self.drop(word_emb + pos_emb[-qlen:])
core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
```

MultiHeadAttn.forward:

```
c = h
head_q = self.q_net(h)
head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)
attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))
```

.bool()

```
| End of training | test loss  2.64 | test bpc   3.81302
bash run_enwik8_small_xl.sh
| End of training | test loss  2.32 | test bpc   3.35011
```

# mine

等长，类rnn

attn_type=4

除了传hidden，还要传相对位置（下一段要根据qlen更新相对位置）

```
bash run_enwik8_small_vanilla.sh
bash run_enwik8_small_xl.sh
bash run_enwik8_small_.sh
```

```
| End of training | test loss  2.64 | test bpc   3.81302
| End of training | test loss  2.32 | test bpc   3.35011
| End of training | test loss  2.39 | test bpc   3.44110
```

可以传递多个段，取mem_len个

取平均

```
| End of training | test loss  2.39 | test bpc   3.44632
```

保存上次prob，累积到上次prob：

不做

虽然是选最大的attn_score，但这是训练，还不是最终最大的，选择策略太差，还是保守的选择最近的较好，延长训练step会不会有效

```
export CUDA_VISIBLE_DEVICES=1

bash run_enwik8_small_vanilla.sh
| End of training | test loss  2.08 | test bpc   3.00053  
scp @172.18.:/data//jiqixuexi/transformer-xl/enwik8_small_vanilla-enwik8/20231218-152919/*.png  D:\data\jiqixuexi\homework\5.1\png\e\vanilla.png

bash run_enwik8_small_xl.sh
| End of training | test loss  1.60 | test bpc   2.30799 
scp @172.18.:/data//jiqixuexi/transformer-xl/enwik8_small_xl-enwik8/20231218-153029/*.png  D:\data\jiqixuexi\homework\5.1\png\e\xl.png

bash run_enwik8_small_.sh
| End of training | test loss  1.63 | test bpc   2.35397
scp @172.18.:/data//jiqixuexi/transformer-xl/enwik8_small_-enwik8/20231218-153152/*.png  D:\data\jiqixuexi\homework\5.1\png\e\.png
```

```
bash run_wt103_small_vanilla.sh  
| End of training | test loss  5.62 | test ppl   276.733
scp @:/data//jiqixuexi/transformer-xl/wt103_small_vanilla-wt103/20231218-134127/*.png  D:\data\jiqixuexi\homework\5.1\png\w\vanilla.png

bash run_wt103_small_xl.sh  
| End of training | test loss  5.45 | test ppl   233.368
scp @.38:/data//jiqixuexi/transformer-xl/wt103_small_xl-wt103/20231218-133953/*.png  D:\data\jiqixuexi\homework\5.1\png\w\xl.png

bash run_wt103_small_.sh  
| End of training | test loss  5.48 | test ppl   238.867
scp @:/data//jiqixuexi/transformer-xl/wt103_small_-wt103/20231218-133703/*.png  D:\data\jiqixuexi\homework\5.1\png\w\.png
```

get_WikiText-103.sh，get_enwik8.sh用于获取实验数据WikiText-103，enwik8

run_enwik8_small_vanilla.sh，run_enwik8_small_xl.sh，run_enwik8_small_.sh分别是用enwik8训练普通transformer，transformer-xl，transformer-re的脚本，可复现表\ref{}的结果

run_wt103_small_vanilla.sh，run_wt103_small_xl.sh，run_wt103_small_.sh分别是用WikiText-103训练普通transformer，transformer-xl，transformer-re的脚本，可复现表\ref{}的结果

train.py是实现训练的主函数，mem_transformer.py实现了普通transformer，transformer-xl，transformer-re

每层模型保存的是输入

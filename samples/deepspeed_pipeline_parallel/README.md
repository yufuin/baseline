# 参考リンク
- https://www.deepspeed.ai/tutorials/pipeline/
- https://deepspeed.readthedocs.io/en/latest/pipeline.html

# チェックリスト
- micro batch sizeは1にする
  - 1でないときはpipeline parallelで学習すると重み更新がおかしくなり発散する。原因不明
    - そもそもpipeline parallelを使うときは限界まで系列長を長くする状況のはずなので、自然とmicro batch sizeは1になるはず
- モデルの定義はtorch.nn.Module 1つにまとめるのではなく、torch.nn.Sequentialか、moduleのリストとして構成する
  - LayerSpecによってメモリの節約が可能、ただし並列数が少ないうちはあまり恩恵がない
  - forwardに複数の引数が必要な場合は最初の入力にあらかじめ仕込んでおいて、前のレイヤーからバケツリレーで渡す必要がある
- バッチサイズ、ミクロバッチサイズ (micro batch size) は設定jsonファイル中の"train_batch_size"と"gradient_accumulation_steps"でコントロールする
  - deepspeed.initializeにデータセットを渡すことで適切なミクロバッチサイズが設定される
  - データセットは勝手にdeepspeed.utils.RepeatingLoaderでラップされ、無限ループする。
    - しかも (deepspeed==0.17.4で試したうちでは) 一番最初のミクロバッチは学習ではスキップされ、次のミクロバッチから学習が始まる。モデルのコンパイルに使われている？
  - データセットの中の総事例数はミクロバッチサイズの約数である必要がある
    - 限界までモデルサイズを大きくするなら必然的にミクロバッチサイズは1になるはずなのでpipeline parallelismを最大限活用する場合はそこまで気にする必要がないかもしれない
  - 無理してinitializeで用意されたdataloaderを使う必要はないかもしれない
    - ただしgradient accumulationの途中でデータセットがstopiterationをraiseすると学習がhaltするらしいのでそうならないように注意する必要がある
    - initialize外のdataloaderのdeviceへの配置方法は未確認
- 損失関数はPipelineModuleのloss_fn引数に指定する
  - 各ミクロバッチごとにpipeline parallel単位で損失関数を一度通過するので、これを利用してミクロバッチ内の損失のログを取れる
- deepspeed用のArgumentParserを用意する
- Datasetは全ワールドで一貫していないとエラーになる
- num_stagesに分割数を指定する
- 学習ループはmodel_engine.train_batchによって実施する
  - 一回のtrain_batchの実行が一回のgradient_accumulation完遂に相当する
  - deepspeed.initializeで用意したtrain_datasetを使う場合はstopiterationが発生しないため、トータルで何回アップデートするかは設定で制御するか自分で管理する必要がある

# サンプルの実行コマンド
## 設定
jsonファイルは自前で用意する必要がある．
accelerateの設定で自動生成する手があるが，意図しない設定がなされる可能性もある．

```json
{
    "train_batch_size": 16,
    "train_micro_batch_size_per_gpu": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.00001
        }
    },

    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 1
    },
    "gradient_clipping": false
}
```

## 実行
### pipeline parallel
```
uv run deepspeed --include localhost:1,2 main.deepspeed.pipe.py --deepspeed --deepspeed_config ./deepspeed.pipe.json --pipeline_size 2 --reduce_memory
```

### non-pipeline parallel
```
uv run deepspeed --include localhost:1,2 main.deepspeed.pipe.py --deepspeed --deepspeed_config ./deepspeed.pipe.json
```

# Llama-3.1のpipeline parallelな実行
## 実装ファイル
- main.deepspeed.pipe.llama.py
- modeling_llama.py
- token_ids.json
- deepspeed.pipe.json

## 設定
- 上記サンプルと同様でok。ただしbfloat16で行う場合はgrad accumulationのマイクロバッチサイズが1になるようにしないと更新時に発散して崩壊する
  - 正確な原因・対処法は現時点では不明

## 実行コマンド
```
uv run deepspeed --include localhost:4,5,6,7 main.deepspeed.pipe.llama.py --deepspeed --deepspeed_config ./deepspeed.pipe.json --pipeline_size 4 --sequence_length 4096 --inherit move
```

## NOTE
- Sequentialの各層間ではただ1つのテンソルの入出力にしないとgradがないというエラーが出て動かない
  - データセットからの入力のみ複数のテンソルが許される。入力側をタプルにして複数のテンソルを並べる。
  - 従って殆どの情報を各層内で個別に再構築するように実装を書き換えないと対応できない
    - catなどで"1つのテンソル"にまとめられるのなら受け渡し可能
  - 現状はmodeling_llama.py内で各層を再実装している
- bfloat16でgrad accumulationのマイクロバッチサイズが2以上だと初回更新時に発散する





# メモ
- 同じGPU数あたりのメモリ効率はZeRO-2とそんなに変わらない？ -> 変わる、が、系列帳が短いうちはoptimizerの使用するメモリのほうが支配的なせいであまり大きなインパクトがない
  - 浅いステージはforwardのキープ数が多いが、深いステージは少ないforwardの保持で済むためうまく分割すればメモリ効率が向上
  - しかしGPUメモリの使用量はパラメータに比べてoptimizer (モデルパラメータをbf16, optimをAdam fp32で想定) が使用する量が6倍程度大きいため、ZeROを使用せずPipeline parallelだけ使用してもメモリ効率は悪い
    - 特にactivationがモデルパラメータ, optimizerに比べてメモリを使っていない場合に顕著
  - 提案論文を読めば書いてあるかも
    - https://huggingface.co/docs/transformers/v4.17.0/en/parallelism
  - Pipeline parallelismはあくまで3D parallelism前提で、単独で使用するものではない
- pipeline有効時gradがないと言われて動かない問題発生 --> 解決
  - 層間で渡せるテンソルは1つだけだった
    - とりあえずLlama-3.1については層間で受け渡していたほとんどの情報は各層で再構築できたので、hidden_statesだけを層間で受け渡してほかはすべて個別に再構築することにした
    - 将来的にattention_maskの受け渡しが必要になるが、どうするべきかは現状不明 --> catでまとめれば一応受け渡せた
      - overheadがどの程度になるか不明
  - 参考
    - https://github.com/deepspeedai/DeepSpeed/issues/7270
    - https://deepspeed.readthedocs.io/en/stable/pipeline.html
    - https://github.com/deepspeedai/DeepSpeed/issues/7270
- モデルがbfloat16でマイクロバッチサイズが2以上だと初回更新時に発散する
  - 現状はgrad accumulationが単発2以上のサイズのマイクロバッチに遭遇したときに更新がおかしくなり初回backward後nanに飛ぶ
    - おそらくampのせい
    - bfloat16だとseq_len=32でも8192でも発生。float32ならseq_len=32では発生しない。
      - float32は8192ではメモリが溢れた
  - こちらで用意した普通のoptimが使えるのであればrandom float32 -> bfloat16のadamを用意すれば解決するはず
    - ただしどの程度fused adamと性能差が出るか不明
      - float32で普通のadamで性能比較をしておきたい

- pipeline parallelなしのzero-2にするときは設定を細かくチューニングすると速度がだいぶ変わった
  - ```json
      "zero_optimization": {
          "stage": 2,
          "contiguous_gradients": true,
          "overlap_comm": true,
          "reduce_scatter": true,
          "reduce_bucket_size": 5e8,
          "allgather_partitions": true,
          "allgather_bucket_size": 5e8
      },
    ```
  - stage 1やpipelineありだとあまり関係なし。おそらくoverlap_commの効果？
    - かわりにロスが悪くなる
  - 計測結果
    - GPU: A100 80G x 4
      - note
        - x4 pipeline parallelならseq_len=8192でも実行可能だが、なしだとstage2でもOOM
        - 下表中ではstageの設定のみのものをstage1 or stage2、追加の設定有りをstage1+ or stage2+と記載
      - table
        - stage1  pipe4 seq=8192 93.2s loss=0.0557
        - stage1+ pipe4 8192 93.4s 0.0615
        - stage2+       8192 OOM
        - stage2        8192 OOM
        - stage1+       8192 OOM
        - stage1        8192 OOM
        - stage2        6144 OOM
        - stage2+       6144 OOM
        - stage1        6144 OOM
        - stage1  pipe4 4096 44.8s 0.0376
        - stage1+ pipe4 4096 45.0s 0.0371
        - stage2        4096 69.4s 0.0383
        - stage2+       4096 48.9s 0.1660
    - 常にpipelineあり, stage1
      - table
        - seq_len=6144 6GPU pipe6 50.7s
        - 6144 4GPU pipe4 68.8s
        - 4096 6GPU pipe3 33.6s
        - 4096 6GPU pipe6 34.2s
        - 4096 4GPU pipe4 44.8s
        - 4096 4GPU pipe2 45.0s

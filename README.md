# UniIGL

UniIGL: Unified Imbalanced Graph Learning with Topology-aware Pretraining and Dual Prompt Fine-tuning

### Running the Code

The code supports three types of imbalance scenarios:

#### 1. Both Class and Topology Imbalance

```bash
python main.py --config "config/Cora_both.yml"
```

#### 2. Class Imbalance Only

```bash
python main.py --config "config/Cora_classnum.yml"
```

#### 3. Topology Imbalance Only

```bash
python main.py --config "config/Cora_topology.yml"
```

# MuZero algorithm adapted to BabyEnv

Note that you need MuZero:  
https://github.com/werner-duvaud/muzero-general

```bash
git clone https://github.com/werner-duvaud/muzero-general.git
cd muzero-general

pip install -r requirements.txt
```

To adapt muZero with BabyEnv, we created a game-adaption class:  
baby.mcts.muzero.baby_game
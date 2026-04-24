#!/bin/bash
#SBATCH -A m4341_g
#SBATCH -t 00:20:00
#SBATCH -C "gpu&hbm40g"
#SBATCH -N 1
#SBATCH -q regular
#SBATCH -o leaderboard.out

# --- Anonymous name generation ---
NAMEFILE="$HOME/.cs5220-leaderboard-name"

generate_name() {
    COLORS="red orange yellow green blue purple pink crimson scarlet gold silver bronze teal coral ivory amber jade violet indigo magenta cyan maroon navy olive salmon"
    ANIMALS="falcon eagle hawk owl raven sparrow wolf tiger lion bear panther fox dolphin whale shark otter seal penguin cobra viper dragon phoenix griffin hydra badger moose bison elk jaguar leopard"

    c=$(echo $COLORS | awk -v seed=$RANDOM 'BEGIN{srand(seed)}{n=split($0,a); print a[int(rand()*n)+1]}')
    a=$(echo $ANIMALS | awk -v seed=$RANDOM 'BEGIN{srand(seed)}{n=split($0,a); print a[int(rand()*n)+1]}')
    n=$(awk -v seed=$RANDOM 'BEGIN{srand(seed); printf "%d", rand()*1000}')
    echo "${c}-${a}-${n}"
}

if [ -f "$NAMEFILE" ]; then
    LEADERBOARD_NAME=$(cat "$NAMEFILE")
else
    LEADERBOARD_NAME=$(generate_name)
    echo "$LEADERBOARD_NAME" > "$NAMEFILE"
fi

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%S")

echo "===== CS5220 HW5 LEADERBOARD SUBMISSION ====="
echo "LEADERBOARD_NAME: $LEADERBOARD_NAME"
echo "TIMESTAMP: $TIMESTAMP"
echo ""

./run.sh

echo ""
echo "===== END CS5220 HW5 LEADERBOARD SUBMISSION ====="

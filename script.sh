$ cat script-array.sh
#!/bin/bash
#SBATCH --job-name=trabajito
#SBATCH -t 0-0:20                                  # tiempo maximo en el cluster (D-HH:MM)
#SBATCH -o slurm-%a.out                            # STDOUT
#SBATCH -e slurm-%a.err                            # STDERR
#SBATCH --mail-type=ALL                            # notificacion cuando el trabajo termine o falle
#SBATCH --mail-user=alonso.valdermann@gmail.com    # mail donde mandar las notificaciones
#SBATCH --chdir=/user/miusuario                    # direccion del directorio de trabajo
#
#SBATCH --ntasks 1                                 # 1 trabajo
#SBATCH --array 1-50%5                           # 100 procesos, 10 simultáneos

python train.py $SLURM_ARRAY_TASK_ID


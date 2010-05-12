#!/bin/bash -e

#PBS -N %(name)s
#PBS -M %(email)s
#PBS -q %(queue)s
#PBS -l nodes=%(nodes)s:ppn=%(ppn)s:myri
#PBS -l walltime=%(walltime)s
#PBS -e stderr
#PBS -o stdout
#PBS -m abe
#PBS -V

cd "%(rundir)s"
mv * /scratch/
cd /scratch/

echo "$( date ): %(name)s started" >> log
%(pre)s
mpiexec -np %(nproc)s %(bin)s
%(post)s
echo "$( date ): %(name)s finished" >> log

mv * "%(rundir)s"


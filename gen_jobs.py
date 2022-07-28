from map_utils import *
from pathlib import Path


def preprocess_pednet_jobs():

    with open('{}/preproces_net.sh'.format(folder), 'w') as the_file:
        the_file.write('#!/bin/bash\n')
        the_file.write('#SBATCH --account={}\n'.format(account))
        the_file.write('#SBATCH --output=preproces_net.out')
        # the_file.write('#SBATCH --mail-user=cheryl.huang@mail.utoronto.ca\n')
        # the_file.write('#SBATCH --mail-type=ALL\n')
        the_file.write('\n')
        the_file.write('module load python/3.7\n')
        the_file.write('module load python scipy-stack\n')

        the_file.write('module load gurobi/9.5.0 python/3.7\n')
        the_file.write('source cpo/bin/activate\n')
        the_file.write('export LD_LIBRARY_PATH=/home/huangw98/build/lib:$LD_LIBRARY_PATH\n')
        the_file.write('export PYTHONPATH=$PYTHONPATH:/home/huangw98/modulefiles/lib/python/\n')
        the_file.write('export LD_LIBRARY_PATH=/home/huangw98/build2/lib:$LD_LIBRARY_PATH\n')
        the_file.write('cd /home/huangw98/projects/def-khalile2/huangw98/walkability/\n')
        the_file.write('python prepare_data.py 1\n')
        the_file.write('deactivate')
    return

def extra_jobs():
    id=1000
    for col in range(1,5):
        for row in range(1,4):
            with open('{}/extra_{}.sh'.format(folder, id), 'w') as the_file:
                the_file.write('#!/bin/bash\n')
                the_file.write('#SBATCH --account={}\n'.format(account))
                the_file.write('#SBATCH --output=preproces_{}.out'.format(id))
                # the_file.write('#SBATCH --mail-user=cheryl.huang@mail.utoronto.ca\n')
                # the_file.write('#SBATCH --mail-type=ALL\n')
                the_file.write('\n')
                the_file.write('module load python/3.7\n')
                the_file.write('module load python scipy-stack\n')

                the_file.write('module load gurobi/9.5.0 python/3.7\n')
                the_file.write('source cpo/bin/activate\n')
                the_file.write('export LD_LIBRARY_PATH=/home/huangw98/build/lib:$LD_LIBRARY_PATH\n')
                the_file.write('export PYTHONPATH=$PYTHONPATH:/home/huangw98/modulefiles/lib/python/\n')
                the_file.write('export LD_LIBRARY_PATH=/home/huangw98/build2/lib:$LD_LIBRARY_PATH\n')
                the_file.write('cd /home/huangw98/projects/def-khalile2/huangw98/walkability/\n')
                the_file.write('python make_more_data.py {} {} {} {}\n'.format(1, row,col,id))
                the_file.write('deactivate')
                id+=1
    return

def preprocess_nia_jobs():
    data_root = "/Users/weimin/Documents/MASC/walkability_data"
    D_NIA = ct_nia_mapping(
        os.path.join(data_root, "neighbourhood-improvement-areas-wgs84/processed_TSNS 2020 NIA Census Tracts.xlsx"))
    all_nias = list(D_NIA.keys())
    for nia in all_nias:
        with open('{}/preproces_nia_{}.sh'.format(folder, nia), 'w') as the_file:
            the_file.write('#!/bin/bash\n')
            the_file.write('#SBATCH --account={}\n'.format(account))
            the_file.write('#SBATCH --output=preproces_nia_{}.out'.format(nia))
            # the_file.write('#SBATCH --mail-user=cheryl.huang@mail.utoronto.ca\n')
            # the_file.write('#SBATCH --mail-type=ALL\n')
            the_file.write('\n')
            the_file.write('module load python/3.7\n')
            the_file.write('module load python scipy-stack\n')

            the_file.write('module load gurobi/9.5.0 python/3.7\n')
            the_file.write('source cpo/bin/activate\n')
            the_file.write('export LD_LIBRARY_PATH=/home/huangw98/build/lib:$LD_LIBRARY_PATH\n')
            the_file.write('export PYTHONPATH=$PYTHONPATH:/home/huangw98/modulefiles/lib/python/\n')
            the_file.write('export LD_LIBRARY_PATH=/home/huangw98/build2/lib:$LD_LIBRARY_PATH\n')
            the_file.write('cd /home/huangw98/projects/def-khalile2/huangw98/walkability/\n')
            the_file.write('python prepare_data.py {} --nia {}\n'.format(1, nia))
            the_file.write('deactivate')

    return

def opt_job():

    #all_models=['GreedySingleDepth']
    all_models = ['OptMultipleDepthCP'] #,'OptMultiple'
    all_31_nia = [2, 3, 5, 6, 21, 22, 24, 25, 26, 27, 28, 43, 44, 55, 61, 72, 85, 91, 110, 111, 112, 113, 115, 121, 124, 125, 135,
     136, 137, 138, 139]

    counter=0

    for model in all_models:
        for nia in all_31_nia:
            #for amenity in ['grocery', 'school', 'restaurant']:
            #for amenity in ['restaurant']:
            for k in range(10):
                with open('{}/{}_{}.sh'.format(folder,model,counter), 'w') as the_file:
                    the_file.write('#!/bin/bash\n')
                    the_file.write('#SBATCH --account={}\n'.format(account))
                    the_file.write('#SBATCH --output=walk_slurms/{}_nia{}_{}.out'.format(model,nia,k))
                    # the_file.write('#SBATCH --mail-user=cheryl.huang@mail.utoronto.ca\n')
                    # the_file.write('#SBATCH --mail-type=ALL\n')
                    the_file.write('\n')
                    the_file.write('module load python/3.7\n')
                    the_file.write('module load python scipy-stack\n')

                    the_file.write('module load gurobi/9.5.0 python/3.7\n')
                    the_file.write('source cpo/bin/activate\n')
                    the_file.write('export LD_LIBRARY_PATH=/home/huangw98/build/lib:$LD_LIBRARY_PATH\n')
                    the_file.write('export PYTHONPATH=$PYTHONPATH:/home/huangw98/modulefiles/lib/python/\n')
                    the_file.write('export LD_LIBRARY_PATH=/home/huangw98/build2/lib:$LD_LIBRARY_PATH\n')
                    the_file.write('cd /home/huangw98/projects/rrg-khalile2/huangw98/walkability\n')
                    #the_file.write('cd /home/huangw98/projects/rrg-khalile2/huangw98/walkability\n')
                    #the_file.write('cd /home/huangw98/scratch/walkability\n')
                    the_file.write('python optimize.py {} {} --cc True --k_array {},{},{} --bp True\n'.format(model,nia,k,k,k))
                    #the_file.write('python optimize.py {} {} --cc True --k {} --amenity {} --bp True\n'.format(model, nia, k, amenity))
                    #the_file.write('python optimize.py {} {} --cc True --k {} --amenity {} \n'.format(model, nia, k, amenity))
                    the_file.write('deactivate')
                counter+=1

    return


def opt_job_extra():
    # id = 1000

    cols = []
    rows=[]
    for col in range(1, 5):
        for row in range(1, 4):
            cols.append(col)
            rows.append(row)
    all_ids = list(range(1000, 1012))

    #'OptMultiple', 'OptMultipleDepth','OptMultipleCP','GreedyMultipleDepth','OptMultipleDepthCP', 'GreedyMultiple'
    all_models = ['OptMultiple','OptMultipleCP','GreedyMultiple'] #,'OptMultiple'
    all_models = ['OptMultiple', 'OptMultipleDepth','OptMultipleCP','GreedyMultipleDepth','OptMultipleDepthCP', 'GreedyMultiple']


    counter=0

    for model in all_models:
        for n in range(len(all_ids)):
            nia =all_ids[n]
            col = cols[n]
            row = rows[n]
            if nia in [1000,1003,1006,1009,1010]:
                for k in range(10):
                    with open('{}/{}_{}.sh'.format(folder,"extra",counter), 'w') as the_file:
                        the_file.write('#!/bin/bash\n')
                        the_file.write('#SBATCH --account={}\n'.format(account))
                        the_file.write('#SBATCH --output=walk_slurms/{}_nia{}_{}.out'.format(model,nia,k))
                        # the_file.write('#SBATCH --mail-user=cheryl.huang@mail.utoronto.ca\n')
                        # the_file.write('#SBATCH --mail-type=ALL\n')
                        the_file.write('\n')
                        the_file.write('module load python/3.7\n')
                        the_file.write('module load python scipy-stack\n')

                        the_file.write('module load gurobi/9.5.0 python/3.7\n')
                        the_file.write('source cpo/bin/activate\n')
                        the_file.write('export LD_LIBRARY_PATH=/home/huangw98/build/lib:$LD_LIBRARY_PATH\n')
                        the_file.write('export PYTHONPATH=$PYTHONPATH:/home/huangw98/modulefiles/lib/python/\n')
                        the_file.write('export LD_LIBRARY_PATH=/home/huangw98/build2/lib:$LD_LIBRARY_PATH\n')
                        the_file.write('cd /home/huangw98/projects/rrg-khalile2/huangw98/walkability\n')
                        #the_file.write('cd /home/huangw98/projects/rrg-khalile2/huangw98/walkability\n')
                        #the_file.write('cd /home/huangw98/scratch/walkability\n')
                        the_file.write('python optimize_extra.py {} {} {} {} --cc True --k_array {},{},{}\n'.format(model,nia,col,row,k,k,k))
                        #the_file.write('python optimize.py {} {} --cc True --k {} --amenity {} --bp True\n'.format(model, nia, k, amenity))
                        #the_file.write('python optimize.py {} {} --cc True --k {} --amenity {} \n'.format(model, nia, k, amenity))
                        the_file.write('deactivate')
                    counter+=1

    return

if __name__ == "__main__":
    folder = 'extra_jobs'
    #account = 'rrg-khalile2'
    account = 'def-khalile2'
    Path(folder).mkdir(parents=True, exist_ok=True)
    opt_job_extra()
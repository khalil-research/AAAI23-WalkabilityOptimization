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


def preprocess_nia_jobs():
    data_root = "/home/huangw98/projects/def-khalile2/huangw98/walkability_data"
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
            the_file.write('python prepare_data.py {} {}\n'.format(1, nia))
            the_file.write('deactivate')

    return

def opt_job():

    #D_NIA=ct_nia_mapping()
    # "./data/neighbourhood-improvement-areas-wgs84/pieces.xlsx"
    #all_nias=D_NIA.keys()
    all_nias=[1109,1099,1030,1031,1062,1024,1051,1110,1117,1120,1062,1057,1067, 1053,1044,1022,1029]
    # try for CP_2
    all_nias=[1022, 1024, 1029, 1030, 1044, 1051, 1053, 1057, 1062, 1064, 1067, 1077, 1099, 1109, 1110, 1114, 1117, 1120]
    # try larger
    #all_nias=[1049, 1050, 1058, 1059, 1065, 1070, 1081, 1082, 1083, 1086, 1094, 1104, 1107, 1123]
    # even larger ,43,44,55,24,112
    # all_pieces:
    all_nias =[1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123]
    all_k=[0,1,2,3]

    infeas=[1032, 1025, 1031, 1069, 1096, 1097, 1054, 1108, 1078, 1079, 1047, 1084, 1085, 1063, 1088, 1089, 1116, 1060, 1074, 1101, 1115, 1111, 1065, 1070, 1072, 1098, 1073, 1015, 1003, 1002]
    probelmatic=[1034,1035,1036,1037,1038,1039,1040,1041,1042,1043,1052,1119,1066]+[1014, 1017, 1016, 1012, 1013, 1007, 1011, 1004, 1009, 1021, 1020, 1008, 1019] + [1083,1121,1055,1122,1075] # second is when m=1 #third is when inf distance in instance
    # too long to run: 1045
    #all_nias=[1045, 1083, 1121, 1055, 1122, 1106, 1105, 1075, 1061, 1010]

    # instances used in the end: [1001, 1005, 1006, 1010, 1018, 1022, 1023, 1024, 1026, 1027, 1028, 1029, 1030, 1033, 1044, 1045, 1046, 1048, 1049, 1050, 1051, 1053, 1056, 1057, 1058, 1059, 1061, 1062, 1064, 1067, 1068, 1071, 1076, 1077, 1080, 1081, 1082, 1086, 1087, 1090, 1091, 1092, 1093, 1094, 1095, 1099, 1100, 1102, 1103, 1104, 1105, 1106, 1107, 1109, 1110, 1112, 1113, 1114, 1117, 1118, 1120, 1123]

    #all_models=['CP_2','CP_1b','CP_1']
    #all_models=['CP_1b','CP_1','CP_2','CP_2b_no_x','MILP']
    all_models=['CP_1b']
    #all_models=['CP_2b_no_x']

    counter=0

    big=[24,28]

    for model in all_models:
        for nia in big:
            if (not nia in infeas) and (not nia in probelmatic):
                with open('{}/job_{}.sh'.format(folder,counter), 'w') as the_file:
                    the_file.write('#!/bin/bash\n')
                    the_file.write('#SBATCH --account={}\n'.format(account))
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
                    the_file.write('cd /home/huangw98/projects/def-khalile2/huangw98/courseProj/\n')
                    the_file.write('python optimize.py {} {} {}\n'.format(model,nia,1))
                    the_file.write('deactivate')
                counter+=1

    return

if __name__ == "__main__":
    folder = 'jobs_walkability'
    account = 'rrg-khalile2'
    Path(folder).mkdir(parents=True, exist_ok=True)
    preprocess_pednet_jobs()
from itertools import product
from datetime import datetime
import os
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff 


def data_loader(dataset_name,task='classification'):

    if task == 'classification':

        print(f'Dataset Name: {dataset_name}')
        if dataset_name == 'mri':
            file_name = f'/home/mxn447/Datasets/MRI'
            thickness_data = pd.read_csv(f'{file_name}/thickness_data.csv', index_col=None)
            X = pd.read_csv(f'{file_name}/X.csv', index_col=False)
            y = pd.read_csv(f'{file_name}/y.csv', index_col=False)
            X = pd.concat([X, thickness_data], axis=1)
            y =  pd.Series((y.values).flatten()) 
        else:
            file_name = f'/home/mxn447/Datasets/classification/{dataset_name}'
            data = loadarff(file_name)
            df = pd.DataFrame(data[0])


        if dataset_name == 'madeline.arff':
            print(f'dataset name is: {dataset_name}')
            y = df['class']
            X = df.drop(columns=['class'])
        elif dataset_name == 'philippine.arff':
            print(f'dataset name is: {dataset_name}')
            y = df['class']
            X = df.drop(columns=['class'])
        elif dataset_name == 'jasmine.arff':
            print(f'dataset name is: {dataset_name}')
            y = df['class']
            X = df.drop(columns=['class'])
        elif dataset_name == 'clean1.arff':
            print(f'dataset name is: {dataset_name}')
            y = df['class']
            X = df.drop(columns=['class', 'conformation_name', 'molecule_name'])
        elif dataset_name == 'clean2.arff':
            print(f'dataset name is: {dataset_name}')
            y = df['class']
            X = df.drop(columns=['class', 'conformation_name', 'molecule_name'])
        elif dataset_name == 'fri_c4_1000_100.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'binaryClass'
            y = df[target_name]
            X = df.drop(columns=[target_name])
        elif dataset_name == 'fri_c4_500_100.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'binaryClass'
            y = df[target_name]
            X = df.drop(columns=[target_name])
        elif dataset_name == 'tecator.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'binaryClass'
            y = df[target_name]
            X = df.drop(columns=[target_name])
        elif dataset_name == 'speech.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'Target'
            y = df[target_name]
            X = df.drop(columns=[target_name])
        elif dataset_name == 'nomao.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'Class'
            y = df[target_name]
            X = df.drop(columns=[target_name])
        elif dataset_name == 'musk.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'class'
            y = df[target_name]
            X = df.drop(columns=[target_name, 'ID'])
        elif dataset_name == 'scene.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'Urban'
            y = df[target_name]
            X = df.drop(columns=[target_name])
        elif dataset_name == 'hill_valley.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'Class'
            y = df[target_name]
            X = df.drop(columns=[target_name])
        elif dataset_name == 'hill_valley_noiseless.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'Class'
            y = df[target_name]
            X = df.drop(columns=[target_name])
        elif dataset_name == 'bioresponse.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'target'
            y = df[target_name]
            X = df.drop(columns=[target_name])
        elif dataset_name == 'eye_movement.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'label'
            y = df[target_name]
            X = df.drop(columns=[target_name])
        elif dataset_name == 'jannis.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'class'
            y = df[target_name]
            X = df.drop(columns=[target_name])
        elif dataset_name == 'miniboone.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'signal'
            y = df[target_name]
            X = df.drop(columns=[target_name])
        elif dataset_name == 'pol_class.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'binaryClass'
            y = df[target_name]
            X = df.drop(columns=[target_name])
        elif dataset_name == 'australian.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'A15'
            y = df[target_name]
            X = df.drop(columns=[target_name])

        elif dataset_name == 'autoUniv-au1-1000.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'Class'
            y = df[target_name]
            X = df.drop(columns=[target_name])
        elif dataset_name == 'climate-model-simulation-crashes.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'Class'
            y = df[target_name]
            X = df.drop(columns=[target_name, 'V1', 'V2'])
        elif dataset_name == 'coil2000.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'CARAVAN'
            y = df[target_name]
            X = df.drop(columns=[target_name])
        elif dataset_name == 'credit-approval.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'class'
            df = df.dropna()
            df = df.replace(b'?', np.nan).dropna()
            df = df.reset_index(drop=True)
            y = df[target_name]
            X = df.drop(columns=[target_name])
        elif dataset_name == 'credit-g.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'class'
            y = df[target_name]
            X = df.drop(columns=[target_name])  
        elif dataset_name == 'ilpd.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'Class'
            y = df[target_name]
            X = df.drop(columns=[target_name])  
        elif dataset_name == 'kc1.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'defects'
            y = df[target_name]
            X = df.drop(columns=[target_name])  
        elif dataset_name == 'kc2.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'problems'
            y = df[target_name]
            X = df.drop(columns=[target_name])  
        elif dataset_name == 'ozone_level.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'Class'
            y = df[target_name]
            X = df.drop(columns=[target_name])  
        elif dataset_name == 'pc1.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'defects'
            y = df[target_name]
            X = df.drop(columns=[target_name])  
        elif dataset_name == 'pc3.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'c'
            y = df[target_name]
            X = df.drop(columns=[target_name])  
        elif dataset_name == 'pc4.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'c'
            y = df[target_name]
            X = df.drop(columns=[target_name]) 
        elif dataset_name == 'qsar-biodeg.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'Class'
            y = df[target_name]
            X = df.drop(columns=[target_name]) 
        elif dataset_name == 'satellite.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'Target'
            y = df[target_name]
            X = df.drop(columns=[target_name]) 
        elif dataset_name == 'spambase.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'class'
            y = df[target_name]
            X = df.drop(columns=[target_name]) 
        elif dataset_name == 'steel-plates-fault.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'Class'
            y = df[target_name]
            X = df.drop(columns=[target_name]) 
        elif dataset_name == 'svmguide3.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'class'
            y = df[target_name]
            X = df.drop(columns=[target_name])
        elif dataset_name == 'w4a.arff':
            print(f'dataset name is: {dataset_name}')
            target_name = 'class'
            y = df[target_name]
            X = df.drop(columns=[target_name])

        if dataset_name != 'mri':
            del df

            for column in X.columns:
                if isinstance(X[column].iloc[0], bytes):
                    X[column] = X[column].str.decode('utf-8')
                    X[column] = X[column].astype('category')

            X = pd.get_dummies(X)


            y= y.str.decode('utf-8')
            if dataset_name == 'pol_class.arff':
                y = y.map({'P': 1, 'N': 0})
            elif dataset_name == 'covertype':
                y = y.map({'2': 1, '1': 0})
            elif dataset_name == 'fri_c4_1000_100.arff':
                y = y.map({'P': 1, 'N': 0})
            elif dataset_name == 'fri_c4_500_100.arff':
                y = y.map({'P': 1, 'N': 0})
            elif dataset_name == 'tecator.arff':
                y = y.map({'P': 1, 'N': 0})
            elif dataset_name == 'speech.arff':
                y = y.map({'Anomaly': 1, 'Normal': 0})
            elif dataset_name == 'nomao.arff':
                y = y.map({'2': 1, '1': 0})  
            elif dataset_name == 'musk.arff':
                y = y.map({'1': 1, '0': 0})  

            elif dataset_name == 'scene.arff':
                y = y.map({'1': 1, '0': 0})  
            elif dataset_name == 'hill_valley.arff':
                y = y.map({'1': 1, '0': 0})  
            elif dataset_name == 'hill_valley_noiseless.arff':
                y = y.map({'1': 1, '0': 0}) 
            elif dataset_name == 'australian.arff':
                y = y.map({'1': 1, '0': 0})  
            elif dataset_name == 'autoUniv-au1-1000.arff':
                y = y.map({'class2': 1, 'class1': 0})  
            elif dataset_name == 'climate-model-simulation-crashes.arff':
                y = y.map({'2': 1, '1': 0})  
            elif dataset_name == 'credit-approval.arff':
                y = y.map({'+': 1, '-': 0}) 
            elif dataset_name == 'credit-g.arff':
                y = y.map({'good': 1, 'bad': 0}) 
            elif dataset_name == 'ilpd.arff':
                y = y.map({'2': 1, '1': 0}) 
            elif dataset_name == 'kc1.arff':
                y = y.map({'true': 1, 'false': 0}) 
            elif dataset_name == 'kc2.arff':
                y = y.map({'yes': 1, 'no': 0}) 
            elif dataset_name == 'pc1.arff':
                y = y.map({'true': 1, 'false': 0}) 
            elif dataset_name == 'pc3.arff':
                y = y.map({'TRUE': 1, 'FALSE': 0}) 
            elif dataset_name == 'pc4.arff':
                y = y.map({'TRUE': 1, 'FALSE': 0}) 
            elif dataset_name == 'qsar-biodeg.arff':
                y = y.map({'2': 1, '1': 0}) 
            elif dataset_name == 'satellite.arff':
                y = y.map({'Anomaly': 1, 'Normal': 0}) 
            elif dataset_name == 'spambase.arff':
                y = y.map({'1': 1, '0': 0}) 
            elif dataset_name == 'steel-plates-fault.arff':
                y = y.map({'2': 1, '1': 0}) 
            elif dataset_name == 'bioresponse.arff':
                y = y.map({'1': 1, '0': 0}) 

            elif dataset_name == 'clean1.arff':
                y = y.map({'1': 1, '0': 0}) 

            elif dataset_name == 'clean2.arff':
                y = y.map({'1': 1, '0': 0}) 
            elif dataset_name == 'jannis.arff':
                y = y.map({'1': 1, '0': 0}) 

            elif dataset_name == 'jasmine.arff':
                y = y.map({'1': 1, '0': 0}) 

            elif dataset_name == 'madeline.arff':
                y = y.map({'1': 1, '0': 0}) 

            elif dataset_name == 'miniboone.arff':
                y = y.map({'True': 1, 'False': 0}) 
            elif dataset_name == 'eye_movement.arff':
                y = y.map({'1': 1, '0': 0}) 
            try:
                print((y == 0).sum()+ (y==1).sum())
                print(y.shape[0])
            except:
                pass


        return X,y
    elif task == 'regression':

        file_name = f'/home/mxn447/Datasets/regression/{dataset_name}'
        data = loadarff(file_name)
        df = pd.DataFrame(data[0])
        for column in df.columns:
            if isinstance(df[column].iloc[0], bytes):
                df[column] = df[column].str.decode('utf-8')
                df[column] = df[column].astype('category')

        df = pd.get_dummies(df)

        if dataset_name == 'lungcancer_shedden.arff':
            y = df.iloc[:,0]
            X = df.iloc[:, 1:]
        elif dataset_name == 'tecator.arff':
            print(f'dataset name is : {dataset_name}')
            y = df['fat']
            X = df.drop(columns=['fat'])
        elif dataset_name == 'residential_building.arff':
            print(f'dataset name is : {dataset_name}')
            y = df['Output_V.9']
            X = df.drop(columns=['Output_V.9'])
            X = X.drop(columns=['Output_V.10'])
        elif dataset_name == 'geographical_origin_music.arff':
            print(f'dataset name is : {dataset_name}')
            y = df['V100']
            X = df.drop(columns=['V100'])
        elif dataset_name == 'mtp.arff':
            print(f'dataset name is : {dataset_name}')
            y = df['oz203']
            X = df.drop(columns=['oz203'])
        elif dataset_name == 'plasma_retinol.arff':
            print(f'dataset name is : {dataset_name}')
            y = df['RETPLASMA']
            X = df.drop(columns=['RETPLASMA'])
        elif dataset_name == 'dataset_autoHorse_fixed.arff':
            print(f'dataset name is : {dataset_name}')
            y = df['price']
            X = df.drop(columns=['price'])

        elif dataset_name == 'liver-disorders.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'drinks'
            y = df[target]
            X = df.drop(columns=[target])

        elif dataset_name == 'bodyfat.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'class'
            y = df[target]
            X = df.drop(columns=[target])

        elif dataset_name == 'meta.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'class'
            y = df[target]
            X = df.drop(columns=[target])
        elif dataset_name == 'CPMP-2015-regression.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'runtime'
            y = df[target]
            X = df.drop(columns=[target, 'instance_id'])

        elif dataset_name == 'boston.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'MEDV'
            y = df[target]
            X = df.drop(columns=[target])

        elif dataset_name == 'boston_corrected.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'LSTAT'
            y = df[target]
            X = df.drop(columns=[target, 'OBS.', 'TOWN_ID'])
        elif dataset_name == 'space_ga.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'ln(VOTES/POP)'
            y = df[target]
            X = df.drop(columns=[target])

        elif dataset_name == 'socmob.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'counts_for_sons_current_occupation'
            y = df[target]
            X = df.drop(columns=[target])

        elif dataset_name == 'stock.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'company10'
            y = df[target]
            X = df.drop(columns=[target])

        elif dataset_name == 'wisconsin.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'time'
            y = df[target]
            X = df.drop(columns=[target])

        elif dataset_name == 'triazines.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'activity'
            y = df[target]
            X = df.drop(columns=[target])
        elif dataset_name == 'diamonds.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'price'
            y = df[target]
            X = df.drop(columns=[target])

        elif dataset_name == 'bike_sharing.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'count'
            y = df[target]
            X = df.drop(columns=[target])


        elif dataset_name == 'medical_charges.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'AverageTotalPayments'
            y = df[target]
            X = df.drop(columns=[target])

        elif dataset_name == 'medical_charges.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'tipamount'
            y = df[target]
            X = df.drop(columns=[target])

        elif dataset_name == 'superconduct.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'criticaltemp'
            y = df[target]
            X = df.drop(columns=[target])

        elif dataset_name == 'cpmp-2015-regression.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'runtime'
            y = df[target]
            X = df.drop(columns=[target])
        elif dataset_name == 'topo_2_1.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'oz267'
            y = df[target]
            X = df.drop(columns=[target])
        elif dataset_name == 'elevators.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'Goal'
            y = df[target]
            X = df.drop(columns=[target])
        elif dataset_name == 'cps_85_wages.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'WAGE'
            y = df[target]
            X = df.drop(columns=[target])

        elif dataset_name == 'cpu_act.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'binaryClass'
            y = df[target]
            X = df.drop(columns=[target])
        elif dataset_name == 'bank32nh.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'rej'
            y = df[target]
            X = df.drop(columns=[target])
        elif dataset_name == 'wind.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'MAL'
            y = df[target]
            X = df.drop(columns=[target])
        elif dataset_name == 'disclosure_x_bias.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'Income'
            y = df[target]
            X = df.drop(columns=[target])
        elif dataset_name == 'rmftsa_ladata.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'Respiratory_Mortality'
            y = df[target]
            X = df.drop(columns=[target])
        elif dataset_name == 'puma32H.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'thetadd6'
            y = df[target]
            X = df.drop(columns=[target])
        elif dataset_name == 'climate-model-simulation-crashes.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'outcome'
            y = df[target]
            X = df.drop(columns=[target])

        elif dataset_name == 'fri_c4_500_10.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'oz11'
            y = df[target]
            X = df.drop(columns=[target])
        elif dataset_name == 'fri_c2_1000_10.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'oz11'
            y = df[target]
            X = df.drop(columns=[target])
        elif dataset_name == 'fri_c2_1000_25.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'oz26'
            y = df[target]
            X = df.drop(columns=[target])
        elif dataset_name == 'fri_c1_500_25.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'oz26'
            y = df[target]
            X = df.drop(columns=[target])
        elif dataset_name == 'fri_c1_1000_50.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'oz51'
            y = df[target]
            X = df.drop(columns=[target])
        elif dataset_name == 'fri_c0_1000_50.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'oz51'
            y = df[target]
            X = df.drop(columns=[target])
        elif dataset_name == 'fri_c4_1000_25.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'oz26'
            y = df[target]
            X = df.drop(columns=[target])
        elif dataset_name == 'fri_c3_1000_25.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'oz26'
            y = df[target]
            X = df.drop(columns=[target])
        elif dataset_name == 'fri_c0_1000_25.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'oz26'
            y = df[target]
            X = df.drop(columns=[target])
        elif dataset_name == 'fri_c4_1000_100.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'oz101'
            y = df[target]
            X = df.drop(columns=[target])
        elif dataset_name == 'fri_c4_500_25.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'oz26'
            y = df[target]
            X = df.drop(columns=[target])
        elif dataset_name == 'fri_c4_500_50.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'oz51'
            y = df[target]
            X = df.drop(columns=[target])
        elif dataset_name == 'fri_c4_500_100.arff':
            print(f'dataset name is : {dataset_name}')
            target = 'oz51'
            y = df[target]
            X = df.drop(columns=[target])
        else:
            X = df.iloc[:, :-1]
            y = df.iloc[:,-1]

        return X,y






        





import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

current_date_folder = datetime.now().strftime("%b %d")
day_folder_path = os.path.join(base_dir, current_date_folder)
if not os.path.exists(day_folder_path):
    os.makedirs(day_folder_path)
os.chdir(day_folder_path)


Classification_LIT_LVM_VS_FMS_REAL = False     #No interaction vs interaction vs litlvm vs FM
Classification_LIT_LVM_VS_FMS_SIM_V_4 = False
Classification_LIT_LVM_VS_FMS_SIM_V_01 = False
Classification_LIT_LVM_vs_EN = False
Regression_LIT_LVM_VS_FMS_V_01 = False
Classification_LIT_VS_EN_SPARSE = False
Regression_LIT_VS_EN_SPARSE = False
Regression_LITLVM_VS_FM_V_4 = False
Classification_LIT_VS_EN_NOISY = False
Classification_LIT_LVM_VS_FMS_SIM_Sparse = False
Classification_FM_L2_VS_FM_EN = False
classification_Diff_LITLVMvsFM = False 
Classification_LIT_EN_TAB_XG = True


if Classification_LIT_LVM_VS_FMS_REAL:
    # Datasets and mean AUC values
    datasets =                              ['bioresponse', 'clean_1', 'clean_2', 'eye_movement', 'fri_c4_500_100', 'fri_c4_1000_100', 'hill_valley', 'jannis', 'jasmine', 'madeline', 'MiniBooNE', 'nomao', 'pol', 'scene', 'tecator']
    mean_auc = {
        'Elastic Net (No Interaction)':     ( 0.791,         0.90 ,     0.944,     0.586,          0.666,            0.672,             0.711,         0.814,    0.831,     0.626,      0.913,       0.985,   0.984, 0.966,   0.969),
        'Elastic Net (with Interaction)':   ( 0.791,         0.922,     0.943,     0.608,          0.576,            0.564,             0.547,         0.826,    0.833,     0.629,      0.911,       0.984,   0.983, 0.968,   0.959),
        'LIT-LVM':                          ( 0.808,         0.947,     0.985,     0.607,          0.666,            0.689,             0.734,         0.827,    0.837,     0.632,      0.911,       0.987,   0.983, 0.967,   0.974),
        'FMs':                               ( 0.785,         0.906,     0.955,     0.585,          0.626,            0.646,             0.705,         0.807,    0.835,     0.621,      0.903,       0.987,   0.954, 0.956,   0.962),

    }

    auc_std_error = {
        'Elastic Net (No Interaction)':     ( 0.013,        0.003,      0.001,     0.004,          0.011,            0.009,             0.02,          0,        0.004,     0.003,      0.001,       0,       0.003, 0.002,   0.004),
        'Elastic Net (with Interaction)':   ( 0.004,        0.004,      0.003,     0.002,          0.02,             0.009,             0.027,         0.001,    0.003,     0.005,      0.002,       0,       0.001, 0.002,   0.006),
        'LIT-LVM':                          ( 0.003,        0.002,      0.001,     0.002,          0.01,             0.008,             0.034,         0.001,    0.003,     0.006,      0.001,       0,       0.003, 0.002,   0.006),
        'FMs':                               ( 0.003,        0.012,      0.001,     0.002,          0.007,            0.005,             0.05,          0.001,    0.005,     0.004,      0.001,       0,       0.002, 0.008,   0.012),
    }

    # Set up the figure size
    plt.figure(figsize=(20, 8))

    # Number of datasets
    n_datasets = len(datasets)

    # Bar width
    bar_width = 0.18

    # Set seaborn style
    sns.set(style="white", context='notebook')

    # Use seaborn color palette
    palette = sns.color_palette("husl", 4)

    # Positions of the bars on the x-axis
    r1 = np.arange(n_datasets)
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]

    # Create bars with specified colors and error bars
    plt.bar(r1, mean_auc['Elastic Net (with Interaction)'], color=palette[0], width=bar_width, edgecolor='grey', label='Elastic Net (with Interaction)', yerr=auc_std_error['Elastic Net (with Interaction)'], capsize=5)
    plt.bar(r2, mean_auc['Elastic Net (No Interaction)'], color=palette[1], width=bar_width, edgecolor='grey', label='Elastic Net (No Interaction)', yerr=auc_std_error['Elastic Net (No Interaction)'], capsize=5)
    plt.bar(r3, mean_auc['FMs'], color=palette[2], width=bar_width, edgecolor='grey', label='FMs', yerr=auc_std_error['FMs'], capsize=5)
    plt.bar(r4, mean_auc['LIT-LVM'], color=palette[3], width=bar_width, edgecolor='grey', label='LIT-LVM', yerr=auc_std_error['LIT-LVM'], capsize=5)

    # Add xticks on the middle of the group bars
    plt.xticks([r + 1.5 * bar_width for r in range(n_datasets)], datasets, rotation=25, ha='right', fontsize=12, fontweight='bold')

    # Add labels
    plt.ylabel('Mean AUC', fontweight='bold', fontsize=14)

    # Add legend
    plt.legend(loc='upper left', fontsize='large', bbox_to_anchor=(0.35, 1), ncol=1, borderaxespad=0.5, frameon=False)

    # Set y-axis limit
    plt.ylim(0.51, 1)

    # Remove extra white space
    plt.tight_layout(pad=2)

    # Adjust x-axis limits to remove extra space
    plt.xlim(-0.5, n_datasets - 0.5 + 4 * bar_width)

    # Display the chart
    plt.savefig('Classification_LIT_LVM_VS_FMS_REAL.pdf', dpi=1000, bbox_inches='tight')
    plt.show()



if Classification_LIT_LVM_VS_FMS_SIM_V_4:
    # true underlying d is 2
    data = [
        [0.8153943759812721, 0.7805363972412027, 0.7643734298060496, 0.8233151183970856, 0.8409644919965897],  #fm20 2
        [0.9212051124522666, 0.874313901201773, 0.9061802833886042, 0.9038283676428419, 0.9226316990285434],   #fm20 5
        [0.9706190596653534, 0.9601542622137587, 0.9680916490804943, 0.9603794845698085, 0.9760037785543028], #fm20 10

        [0.9930051116491794, 0.9950316059734476, 0.9897993471582182, 0.9932229128550599, 0.9909863207471348],   #lit_lvm 20 2
        [0.9843197250229884, 0.9865654300524871, 0.9751424091141833, 0.9843597914369605, 0.9800437221713202], #lit_lvm 20 5
        [0.9814607350655922, 0.9853101285363753, 0.9744903673835126, 0.985076676400147, 0.977451736831107],#lit_lvm 20 10 


        [0.7385458545854585, 0.7195664269045875, 0.7660976088358142, 0.7335842258032229, 0.7786299621565516],     #fm30 2
        [0.8187338733873387, 0.8156033478631555, 0.823972512224131, 0.8006205585026525, 0.8501205578374711],      #fm30 5
        [0.8806280628062807, 0.8864232220981383, 0.8856357949789957, 0.8918786908217397, 0.9083316168482103],    #fm30 10


[0.9571597159715972, 0.9535766294514805, 0.9660346242304941, 0.9534640868062223, 0.9621510673234812], #lit_lvm 30 2
[0.9563796379637964, 0.949143703038059, 0.9619925707056213, 0.9505033288521861, 0.9649384236453202], #lit_lvm 30 5
[0.961000100010001, 0.9522648041689568, 0.966608595831026, 0.9526638819537802, 0.9620771756978654], #lit_lvm 30 10

        [0.7216740973966407, 0.7476905940893737, 0.7319248995549574, 0.7420994943676396, 0.7347726110366102], #fm 40 2
        [0.7857065602730342, 0.776497128628562, 0.7781115931404948, 0.7580725166410651, 0.7765213294190135], #fm 40 5
        [0.8578307075416903, 0.8352280752731132, 0.8194312587376171, 0.8378296210957502, 0.8189586781518934], #fm 40 10


[0.9398932057891597, 0.9330906527946307, 0.9067281223993342, 0.91404664197778, 0.9269075050758677], # lit_lvm 40 2
[0.9394405521573151, 0.9295385465598232, 0.9031352026118686, 0.9160364327895105, 0.9241763478581241], # lit_lvm 40 5
[0.9399973561823274, 0.9334991249884868, 0.9049196594328147, 0.9137423681313183, 0.9225464637120511], # lit_lvm 40 10

        [0.697526566801101, 0.6976383133216458, 0.6934563974037659, 0.7303438721250559, 0.7065359999999999], #fm 50 2
        [0.7552173356379234, 0.7131395148252304, 0.738504916136495, 0.7524104683195592, 0.708342], #fm 50 5
        [0.780395781320018, 0.7294229974605951, 0.7960044341623289, 0.788655583317317, 0.7626000000000001], #fm 50 10


[0.873915722424941, 0.87306231788321, 0.8891611023042065, 0.8797867571721311, 0.8976916900843035], #lit_lvm 50 2
[0.8693305486204469, 0.8680289036315963, 0.8813429728907463, 0.8763167584528688, 0.8921999197109595], #lit_lvm 50 5
[0.8579716407400294, 0.8681707470283808, 0.8781404621223039, 0.8790183145491803, 0.8833801686069852], #lit_lvm 50 10

        [0.7041359342650104, 0.6823209959731114, 0.7281420160254991, 0.6401041750012002, 0.7382209200495579], #fm 60 2
        [0.7341849443581782, 0.7208531870176957, 0.740326144261688, 0.7212758637243762, 0.7251554737661982], #fm 60 5
        [0.7685081845238095, 0.7048218382506297, 0.7549370369326986, 0.7383743259029301, 0.7638453374012777], #fm 60 10

[0.8341178183229814, 0.8529394030750678, 0.8224808562303905, 0.8311567761346904, 0.8017731649268138], #lit_lvm 60 2
[0.8366491977225673, 0.8254085016581248, 0.8210603504582084, 0.8285761154855641, 0.8032380929446659], #lit_lvm 60 5
[0.8312952898550725, 0.8295769269420159, 0.8216053117699622, 0.8214142820562065, 0.8025598112048933], #lit_lvm 60 10
    ]

    grouped_data = [data[i:i+3] for i in range(0, len(data), 3)]

    # Calculate means and standard deviations for each group
    means = []
    std_devs = []
    for group in grouped_data:
        group_means = [np.mean(region) for region in group]
        group_std_devs = [np.std(region) for region in group]
        means.append(group_means)
        std_devs.append(group_std_devs)

    # Plotting
    #sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(grouped_data))  # the label locations
    width = 0.2  # the width of the bars

    # Labels for the legends as per the user's specification
    legend_labels = ['d:2', 'd:5', 'd:10']
    colors = sns.color_palette("viridis", n_colors=3)
    for i in range(3):  # Three bars per group, adjust for each dimension
        means_plot = [group[i] for group in means]
        std_devs_plot = [group[i] for group in std_devs]
        rects = ax.bar(x - width + i * width, means_plot, width, label=legend_labels[i], 
                    yerr=std_devs_plot, capsize=5, color=colors[i])

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean AUC', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['p:20, FM', 'p:20, LIT_LVM', 'p:30, FM', 'p:30, LIT_LVM', 'p:40, FM', 'p:40, LIT_LVM', 'p:50, FM', 'p:50, LIT_LVM', 'p:60, FM', 'p:60, LIT_LVM'], rotation=35,fontsize=12, fontweight='bold',  ha="right")
    ax.legend()
    ax.set_ylim(0.65, 1)
    fig.tight_layout()
    plt.savefig('LIT_LVM_VS_FMs_V_NOISE_4.pdf', dpi=1000, bbox_inches='tight')
    plt.show()


if Classification_LIT_LVM_VS_FMS_SIM_V_01:

    # true underlying d is 2
    data = [
 [0.9956843250729583, 0.9944019239831652, 0.9959007187780773, 0.9963207937164117, 0.9923477579139195],   #fm20 2
 [0.991743747011591, 0.9918344464485935, 0.9912571705024535, 0.9923522116577097, 0.9883110438325431],    #fm 20 5
 [0.9883431435590508, 0.9898763385820375, 0.9885401548137396, 0.990293509714758, 0.9877939440948461],     #fm 20 10     


 [0.9978565893390051, 0.9960715968735273, 0.9969313985541166, 0.997304638181939, 0.9986599582367353], #lit 20 2
 [0.997052810341132, 0.9914363249321567, 0.9944509123788262, 0.9965653864044123, 0.9964361465989248], #lit 20 5
[0.9951113749155001, 0.9872621427062515, 0.9942268684662194, 0.9942105787759985, 0.9949234614183069], #lit 20 10

 [0.9937023407564779, 0.989989928010039, 0.9902103332973788, 0.9932521730290245, 0.9947435107184439], #fm 30 2
 [0.9882435263519522, 0.978663067168615, 0.9809988135044763, 0.9806460631322117, 0.990608519964386], #fm 30 5
 [0.9785609806475545, 0.9687809589855358, 0.9677316362851904, 0.9689972720649284, 0.9782720361619067], #fm 30 10


[0.9946170610080469, 0.9971284127128283, 0.9965969698594651, 0.9971148879341842, 0.9980559067528014], # lit 30 2
[0.9921721866029313, 0.9934135128940325, 0.9915481105964172, 0.9923273657289002, 0.9926392906008071], # lit 30 5
[0.9870000126459021, 0.9860183106936353, 0.9905251391117473, 0.9913110671554979, 0.9895607821403211], # lit 30 10

 [0.9890482541579189, 0.9860896445131376, 0.9758965938953327, 0.9889916666666667, 0.9917763948029261], #fm 40 2
 [0.9791946280487549, 0.9710408904028618, 0.9759765732445802, 0.9768166666666667, 0.9795960257670051], #fm 40 5
 [0.9622726450380554, 0.9476686909729367, 0.9561933215127523, 0.9604750000000001, 0.9574320340648542], #fm 40 10

  [0.9953003841315388, 0.9937759940715822, 0.9928115015974441, 0.9961004143309773, 0.9931220799821546], # lit 40 2
 [0.9840562270962095, 0.9883887942914, 0.9841878662589054, 0.988502208482454, 0.9915440827167992], # lit 40 5
 [0.9762494793397818, 0.9819649540589261, 0.9503297397960055, 0.9767072445921042, 0.973744108328273], # lit 40 10


 [0.9870079200967579, 0.9856511031813842, 0.9814011026642011, 0.9866572686353069, 0.9888626937530816], #fm 50 2
 [0.9689048417931411, 0.9703556650081253, 0.956800904047668, 0.9627005325551103, 0.9707189943277632], #fm 50 5
 [0.9352520168719303, 0.934001775812099, 0.9327015272926512, 0.9442589569833968, 0.9509262443495158], #fm 50 10


[0.9926868072082852, 0.991671857146982, 0.9914476799242424, 0.9931876253827474, 0.9943609022556391], # lit 50 2
[0.9754442320280563, 0.9804315454706699, 0.9810310132575758, 0.9839277795375356, 0.9862630907626209], # lit 50 5
[0.9450101656414325, 0.9709958709900961, 0.9640447443181819, 0.947274838982156, 0.9423335123523093], # lit 50 10

[0.9867300940285424, 0.9880521923911993, 0.9821369225656364, 0.9887478236886108, 0.9857250963126032], #fm 60 2 
[0.9581788811804965, 0.9618552068503562, 0.9602639450070839, 0.9611323714467946, 0.9546341840946615], #fm 60 5 
[0.9343357704510967, 0.9146225206724368, 0.9097488237043256, 0.9232947199669028, 0.9068605531095211], #fm 60 10 


[0.9936987971644587, 0.9931209083469722, 0.9916803940165394, 0.994368, 0.9920762517225539], # lit 60 2
[0.9794276407424384, 0.9816642798690671, 0.980224296577314, 0.9642922666666667, 0.9709802820735296], # lit 60 3
 [0.9449401252230564, 0.9314392389525368, 0.9377943060616648, 0.9316138666666667, 0.9278653941033361], # lit 60 4
    ]

    grouped_data = [data[i:i+3] for i in range(0, len(data), 3)]

    # Calculate means and standard deviations for each group
    means = []
    std_devs = []
    for group in grouped_data:
        group_means = [np.mean(region) for region in group]
        group_std_devs = [np.std(region) for region in group]
        means.append(group_means)
        std_devs.append(group_std_devs)

    # Plotting
    #sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(grouped_data))  # the label locations
    width = 0.25  # the width of the bars

    # Labels for the legends as per the user's specification
    legend_labels = ['d:2', 'd:5', 'd:10']
    colors = sns.color_palette("husl", n_colors=3)
    for i in range(3):  # Three bars per group, adjust for each dimension
        means_plot = [group[i] for group in means]
        std_devs_plot = [group[i] for group in std_devs]
        rects = ax.bar(x - width + i * width, means_plot, width, label=legend_labels[i], 
                    yerr=std_devs_plot, capsize=5, color=colors[i])

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean AUC', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['p:20, FM', 'p:20, LIT_LVM', 'p:30, FM', 'p:30, LIT_LVM', 'p:40, FM', 'p:40, LIT_LVM', 'p:50, FM', 'p:50, LIT_LVM', 'p:60, FM', 'p:60, LIT_LVM'], rotation=25, fontsize=12, fontweight='bold' , ha="right")
    ax.legend(loc='lower left', fontsize=14, markerscale=2)
    ax.set_ylim(0.9, 1)
    fig.tight_layout()
    plt.savefig('LIT_LVM_VS_FMs_V_NOISE_0.1.pdf', dpi=1000, bbox_inches='tight')
    plt.show()



if Classification_LIT_LVM_vs_EN:

    # true underlying d is 2
    data = [


[0.9541124999999999, 0.9633237333333333, 0.9667461444805194, 0.9644703276047263, 0.9756162915326902],  #lit 100 2
[0.9285958333333333, 0.9345877333333334, 0.9466484036796536, 0.93937550349087, 0.949837527007945], #lit 100 5
[0.8766250000000001, 0.8853034666666667, 0.8755580357142857, 0.8724867414070892, 0.9016783204886099],  #lit 100 10
[0.7067958333333333, 0.6868823000898473, 0.703678108927644, 0.6818111773166221, 0.7191639529301866],    # elasticnet


[0.7929766772843616, 0.8245376, 0.8173414281783159, 0.7852461799660442, 0.8032640067911714], # lit 200 2
[0.7561131414615917, 0.8003967999999999, 0.7964966290588883, 0.786774193548387, 0.7872410865874363], # lit 200 5
[0.696161551197424, 0.7468117333333333, 0.7573567350027518, 0.7602461799660442, 0.7355135823429542], # lit 200 10
[0.5783051377765417, 0.6215596330275229, 0.6083043822234452, 0.6082725689955353, 0.6192924577238303],  # elastic net



 [0.5343702721097282, 0.5620834515031197, 0.5193779493779495, 0.5376327423152584, 0.5174744200662044], # lit 500 2      
 [0.5329917331009759, 0.5590169455473624, 0.5246246246246247, 0.5282918157125577, 0.4822156505718627], # lit 500 2         
 [0.49412386909948447, 0.5657348979013046, 0.5201287001287002, 0.5060027291002148, 0.531494419119198], # lit 500 10
[0.5309065844745295, 0.5468671485447025, 0.5097924173894768, 0.5258268448372615, 0.5222531649894927],  # elasticnet                                                                                         # elasticnet   
    ]
 
    # Assuming 'data' is defined somewhere in your script
    step = 4
    grouped_data = [data[i:i+step] for i in range(0, len(data), step)]

    # Calculate means and standard deviations for each group
    means = []
    std_devs = []
    for group in grouped_data:
        group_means = [np.mean(region) for region in group]
        group_std_devs = [np.std(region) for region in group]
        means.append(group_means)
        std_devs.append(group_std_devs)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = np.arange(len(grouped_data))  # the label locations
    width = 0.2  # the width of the bars

    # Calculate the offset to center the bars
    offset = width * (step - 1) / 2

    # Labels for the legends
    legend_labels = ['d:2', 'd:5', 'd:10', 'elastic net']
    colors = sns.color_palette("husl", n_colors=step)
    for i in range(step):
        means_plot = [group[i] for group in means]
        std_devs_plot = [group[i] for group in std_devs]
        rects = ax.bar(x - offset + i * width, means_plot, width, label=legend_labels[i], 
                    yerr=std_devs_plot, capsize=5, color=colors[i])

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean AUC', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['p:100', 'p:200', 'p:500'],rotation=25, fontsize=12, fontweight='bold' , ha="right")
    ax.legend()
    ax.set_ylim(0.4, 1)
    fig.tight_layout()
    plt.savefig('LIT_LVM_VS_ElasticNet.pdf', dpi=1000, bbox_inches='tight')
    plt.show()


if Classification_LIT_EN_TAB_XG:

    # true underlying d is 2
    data = [


[0.9541124999999999, 0.9633237333333333, 0.9667461444805194, 0.9644703276047263, 0.9756162915326902],  #lit 100 2
[0.946, 0.934, 0.895, 0.929, 0.938], #tab
[0.8766250000000001, 0.8853034666666667, 0.8755580357142857, 0.8724867414070892, 0.9016783204886099],  #xg
[0.7067958333333333, 0.6868823000898473, 0.703678108927644, 0.6818111773166221, 0.7191639529301866],    # elasticnet


[0.7929766772843616, 0.8245376, 0.8173414281783159, 0.7852461799660442, 0.8032640067911714], # lit 200 2
[0.7561131414615917, 0.8003967999999999, 0.7964966290588883, 0.786774193548387, 0.7872410865874363], # tab
[0.696161551197424, 0.7468117333333333, 0.7573567350027518, 0.7602461799660442, 0.7355135823429542], # xg
[0.5783051377765417, 0.6215596330275229, 0.6083043822234452, 0.6082725689955353, 0.6192924577238303],  # elastic net



 [0.5343702721097282, 0.5620834515031197, 0.5193779493779495, 0.5376327423152584, 0.5174744200662044], # lit 500 2      
 [0.5329917331009759, 0.5590169455473624, 0.5246246246246247, 0.5282918157125577, 0.4822156505718627], # tab        
 [0.49412386909948447, 0.5657348979013046, 0.5201287001287002, 0.5060027291002148, 0.531494419119198], # xg
[0.5309065844745295, 0.5468671485447025, 0.5097924173894768, 0.5258268448372615, 0.5222531649894927],  # elasticnet                                                                                         # elasticnet   
    ]
 
    # Assuming 'data' is defined somewhere in your script
    step = 4
    grouped_data = [data[i:i+step] for i in range(0, len(data), step)]

    # Calculate means and standard deviations for each group
    means = []
    std_devs = []
    for group in grouped_data:
        group_means = [np.mean(region) for region in group]
        group_std_devs = [np.std(region) for region in group]
        means.append(group_means)
        std_devs.append(group_std_devs)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = np.arange(len(grouped_data))  # the label locations
    width = 0.2  # the width of the bars

    # Calculate the offset to center the bars
    offset = width * (step - 1) / 2

    # Labels for the legends
    legend_labels = ['d:2', 'd:5', 'd:10', 'elastic net']
    colors = sns.color_palette("husl", n_colors=step)
    for i in range(step):
        means_plot = [group[i] for group in means]
        std_devs_plot = [group[i] for group in std_devs]
        rects = ax.bar(x - offset + i * width, means_plot, width, label=legend_labels[i], 
                    yerr=std_devs_plot, capsize=5, color=colors[i])

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean AUC', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['p:100', 'p:200', 'p:500'],rotation=25, fontsize=12, fontweight='bold' , ha="right")
    ax.legend()
    ax.set_ylim(0.4, 1)
    fig.tight_layout()
    plt.savefig('LIT_LVM_VS_ElasticNet.pdf', dpi=1000, bbox_inches='tight')
    plt.show()

if Regression_LIT_LVM_VS_FMS_V_01:
    data = [
[6459.235, 6322.176, 8074.757, 4098.4575, 5270.4644], 
[8566.611, 8055.5684, 10208.612, 5508.896, 7035.129],
[12879.89, 11938.124, 14278.738, 8218.813, 10083.732],  #fm 40      2,5,10


[0.0015902249, 0.0018207575, 0.0016229472, 0.0017559192, 0.00175217], #lit 40
[0.0016705856, 0.0018584437, 0.0016470655, 0.0017786856, 0.0019566051], #lit 40
[0.0016819251, 0.0018231783, 0.0016418302, 0.0017723584, 0.0017557279], #lit 40

[9442.874, 9074.898, 3083.6814, 7058.0645, 11570.004],
[12368.147, 11749.229, 4711.622, 9257.26, 14822.144],
[17520.287, 16922.805, 8259.173, 13532.593, 20559.684], # fm 50 

[0.004760353, 0.0047800196, 0.0044474523, 0.00435295, 0.003960501],
[0.005482502, 0.006005022, 0.0052149477, 0.005107863, 0.005065779], #lit 50
[0.005925469, 0.006863554, 0.0061674863, 0.0059911166, 0.005800886], #lit 50


[21626.568, 12924.907, 15554.126, 12872.068, 18038.922], 
[26962.527, 16919.951, 19578.91, 16413.414, 22269.996], 
[57969.71, 25240.76, 29206.617, 23240.035, 31183.818], # fm 60


[7.5102515, 0.38026625, 0.45975372, 0.31924742, 8.800584], # lit 60
[7.300456, 9.767184, 0.23632962, 0.37339136, 8.675177], # lit 60
[0.983957, 10.301063, 0.23249231, 0.4924709, 8.334006],   # lit 60

[13805.03, 18711.115, 15840.496, 20455.664, 20534.637], 
[19357.936, 24346.863, 20440.56, 25242.412, 25669.283], 
[27102.572, 34066.387, 30073.389, 35973.62, 36910.258],  #fm 70 2,5,10


[70.31523, 80.71183, 487.64273, 67.548836, 95.98661], #lit70
[503.5755, 138.44402, 568.3584, 628.0259, 90.31407], #lit70
[610.9707, 673.49335, 895.26935, 190.7809, 202.62535], #lit70

[29916.674, 27465.922, 30112.002, 43879.047, 33732.86], #fm100
[41316.56, 37933.152, 41523.97, 56030.746, 45015.977],#fm100
[65757.125, 60394.09, 66475.82, 86500.03, 72315.89],#fm100

[3108.0999, 4881.08, 4667.4937, 4815.5073, 3913.5645],   #lit 100
[3182.8625, 5364.4414, 5479.5645, 4605.26, 4155.381],  #lit 100
[3718.3936, 5377.497, 6893.655, 5563.215, 5367.0303],   #lit 100

[139673.55, 148844.77, 139335.53, 113724.45, 136771.42],  #fm200
[214791.17, 225063.3, 208306.97, 165381.39, 203984.53],   # fm200
[352929.16, 357562.06, 342327.16, 280288.72, 333493.47], #fm200

[48427.074, 52218.043, 30532.977, 52082.375, 56008.742], #lit 200
[47166.543, 50252.664, 47757.68, 52929.01, 53673.42], #lit 200
[48647.55, 35157.355, 47503.09, 53020.5, 53811.312], #lit 200
    ]

    data = [[np.sqrt(element) for element in group] for group in data]

    grouped_data = [data[i:i+3] for i in range(0, len(data), 3)]

    # Calculate means and standard deviations for each group
    means = []
    std_devs = []
    for group in grouped_data:
        group_means = [np.mean(region) for region in group]
        group_std_devs = [np.std(region) for region in group]
        means.append(group_means)
        std_devs.append(group_std_devs)

    # Plotting
    #sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(grouped_data))  # the label locations
    width = 0.2  # the width of the bars

    # Labels for the legends as per the user's specification
    legend_labels = ['d:2', 'd:5', 'd:10']
    colors = sns.color_palette("husl", n_colors=3)
    for i in range(3):  # Three bars per group, adjust for each dimension
        means_plot = [group[i] for group in means]
        std_devs_plot = [group[i] for group in std_devs]
        rects = ax.bar(x - width + i * width, means_plot, width, label=legend_labels[i], 
                    yerr=std_devs_plot, capsize=5, color=colors[i])

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean RMSE', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['p:40, FM', 'p:40, LIT_LVM', 'p:50, FM', 'p:50, LIT_LVM', 'p:60, FM', 'p:60, LIT_LVM', 'p:70, FM', 'p:70, LIT_LVM', 'p:100, FM', 'p:100, LIT_LVM', 'p:200, FM', 'p:200, LIT_LVM'], rotation=35, fontsize=12, fontweight='bold' , ha="right")
    ax.legend()
    ax.set_yscale('log')
    #ax.set_ylim(0.9, 1)
    fig.tight_layout()
    plt.savefig('Regression_LIT_LVM_VS_FMs_V_NOISE_0.1.pdf', dpi=1000, bbox_inches='tight')
    plt.show()


if Classification_LIT_VS_EN_SPARSE:
    data = [
[0.9908839868225567, 0.9827613867898798, 0.982304667809255, 0.9846335839598999, 0.9919814019742397],  #lit 20
[0.9916591305471223, 0.9855689785179125, 0.984300836777901, 0.9876571428571428, 0.9937779840668377],  #lit 20
[0.9922203023060526, 0.9850997096147984, 0.9819054340155258, 0.9883829573934837, 0.9939260320343791],  #lit 20
[0.9898383502357729, 0.9881197701381188, 0.9806915132465199, 0.9838105784949639, 0.9862747474747474],  #en 20

[0.9696361947767049, 0.965658914728682, 0.9707572115384615, 0.9693499087866815, 0.9655219220849736], #lit 30
[0.9708051112071865, 0.9673888208894329, 0.9730528846153845, 0.9693499087866817, 0.9693596752100891],#lit 30
[0.9721421594530112, 0.9699796001631986, 0.9730729166666666, 0.9695351543780379, 0.9693308817782823], #lit 30
[0.9625546428399865, 0.9579021119007106, 0.9522342995169083, 0.9663149432217867, 0.9615744324045407],  # en30

[0.9454994440774102, 0.9145664160401002, 0.9344427331504389, 0.9493151890430247, 0.9390032274282748], # lit 40
[0.9421397379912665, 0.9101914786967418, 0.9142521377805587, 0.9539552632842125, 0.9446593360255598], # lit 40
[0.9427440016758247, 0.9192140350877193, 0.9345991561181434, 0.9504552072833166, 0.9398222481713943], # lit 40
[0.9333497155932258, 0.9307034829599719, 0.9223994905058027, 0.9158615136876007, 0.9250600654381256],  # en 40

[0.8851465438382274, 0.8914863159937888, 0.9050786967418546, 0.9110567263024224, 0.8858481619006284], #lit 50
[0.8917151250649806, 0.8975114842132506, 0.9065784461152883, 0.9121417434215813, 0.8875440034720548], #lit 50
[0.8937461464995103, 0.8930552859730848, 0.9021754385964913, 0.9106066451270676, 0.8920850010448313], #lit 50
[0.8792630293651849, 0.8793183105608029, 0.8899960193063642, 0.9257752496159753, 0.8997252525252525],  # en 50

[0.8463146261772053, 0.8277204515991797, 0.8550895115938059, 0.8434045112781955, 0.8697758251661597], #lit 60
[0.8513918252290347, 0.829763323821335, 0.8535143975893908, 0.8432080200501253, 0.8653663560726759], #lit 60
[0.8504949067845474, 0.8304897675781328, 0.8551660516605166, 0.8447318295739349, 0.8678486940568726],#lit 60
[0.8471554872189121, 0.8242864774732481, 0.8376646464646464, 0.8308299306500746, 0.8177663936890459], # en 60
]
    step = 4
    grouped_data = [data[i:i+step] for i in range(0, len(data), step)]

    # Calculate means and standard deviations for each group
    means = []
    std_devs = []
    for group in grouped_data:
        group_means = [np.mean(region) for region in group]
        group_std_devs = [np.std(region) for region in group]
        means.append(group_means)
        std_devs.append(group_std_devs)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(grouped_data))  # the label locations
    width = 0.2  # the width of the bars

    # Calculate the offset to center the bars
    offset = width * (step - 1) / 2

    # Labels for the legends
    legend_labels = ['d:2', 'd:5', 'd:10', 'elastic net']
    colors = sns.color_palette("husl", n_colors=step)
    for i in range(step):
        means_plot = [group[i] for group in means]
        std_devs_plot = [group[i] for group in std_devs]
        rects = ax.bar(x - offset + i * width, means_plot, width, label=legend_labels[i], 
                    yerr=std_devs_plot, capsize=5, color=colors[i])

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean AUC', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['p:20', 'p:30', 'p:40','p:50', 'p:50' ],rotation=35, fontsize=12, fontweight='bold' , ha="right")
    ax.legend()
    ax.set_ylim(0.4, 1)
    fig.tight_layout()
    plt.savefig('Classification_LIT_VS_EN_SPARSE.pdf', dpi=1000, bbox_inches='tight')
    plt.show()

if Regression_LIT_VS_EN_SPARSE: 
    data = [
[0.0015843167, 0.0018603604, 0.0016213832, 0.0017978066, 0.0017482447],  #lit 40 
[0.0016352277, 0.0018347132, 0.0016230418, 0.001758791, 0.001742423],  #lit 40 
[0.0016128214, 0.0018736874, 0.0016259496, 0.0017698687, 0.0017461845],  #lit 40 
[0.0016544202, 0.0018469172, 0.001617534, 0.0017592253, 0.0017848412], # en 40

[1.0350007, 0.891886, 1.1116976, 0.4900337, 1.0800306], #lit60
[1.0705147, 0.45682356, 1.4352192, 0.31990466, 0.8078193],  #lit60
[0.80077845, 0.70749867, 0.60336053, 0.44238707, 0.6969747],  #lit60
[0.66011256, 0.564507, 0.5763606, 0.23563385, 0.3754255],      # en 60


[215.41316, 146.90477, 228.19128, 143.46323], #lit70
[214.42303, 224.79767, 123.7925, 188.94446, 127.573875],#lit70
[219.2061, 199.20728, 123.55028, 201.67035, 156.86305], #lit70
[119.94688, 197.27547, 357.277, 265.34024],  # en 70


[1830.0695, 2407.626, 2326.2498, 3224.7712, 2319.2822], # lit 100
[1730.6703, 2395.4763, 2499.647, 3139.691, 2396.612], # lit 100
[1926.9388, 2739.517, 2565.818, 3621.6323, 2471.622], # lit 100
[2308.8162, 3214.8403, 2791.6897, 2685.8154, 3111.892], # en100

]
    step = 4

    data = [[np.sqrt(element) for element in group] for group in data]
    grouped_data = [data[i:i+step] for i in range(0, len(data), step)]

    # Calculate means and standard deviations for each group
    means = []
    std_devs = []
    for group in grouped_data:
        group_means = [np.mean(region) for region in group]
        group_std_devs = [np.std(region) for region in group]
        means.append(group_means)
        std_devs.append(group_std_devs)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(grouped_data))  # the label locations
    width = 0.2  # the width of the bars

    # Calculate the offset to center the bars
    offset = width * (step - 1) / 2

    # Labels for the legends
    legend_labels = ['d:2', 'd:5', 'd:10', 'elastic net']
    colors = sns.color_palette("husl", n_colors=step)
    for i in range(step):
        means_plot = [group[i] for group in means]
        std_devs_plot = [group[i] for group in std_devs]
        rects = ax.bar(x - offset + i * width, means_plot, width, label=legend_labels[i], 
                    yerr=std_devs_plot, capsize=5, color=colors[i])

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean RMSE', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['p:40','p:60', 'p:70', 'p:100' ],rotation=35, fontsize=12, fontweight='bold' , ha="right")
    ax.legend()
    ax.set_yscale('log')
    fig.tight_layout()
    plt.savefig('Regression_LIT_VS_EN_SPARSE.pdf', dpi=1000, bbox_inches='tight')
    plt.show()


if Classification_LIT_LVM_VS_FMS_SIM_V_4:
    # true underlying d is 2
    data = [

[14773.154, 11556.514, 13338.994, 10336.759, 10544.581],# fm40
[37407.36, 27867.99, 33414.062, 26516.57, 28395.03], # fm40
[163516.31, 85978.59, 114898.35, 72650.734, 94792.09], # fm40

[18206.564, 17502.365, 12484.413, 14744.548, 22378.115],  # fm 50
[48881.773, 45667.29, 34765.656, 38340.49, 51273.383], # fm 50
[203213.4, 113932.4, 98844.875, 105387.13, 287052.88],  # fm 50




[35479.543, 23212.084, 27249.66, 23687.428, 31320.896],   #fm60
[92656.35, 66242.52, 65267.137, 60523.855, 80748.75],  #fm60
[204970.0, 165473.4, 155492.12, 158448.1, 183537.5],  #fm60

[32360.924, 33910.348, 31758.148, 41328.84, 39102.08],  ## fm 70
[87597.34, 92817.35, 83748.414, 99596.49, 91460.625], # fm 70
[224637.27, 232545.47, 213323.22, 233740.42, 221266.44], # fm 70


    ]

    grouped_data = [data[i:i+3] for i in range(0, len(data), 3)]

    # Calculate means and standard deviations for each group
    means = []
    std_devs = []
    for group in grouped_data:
        group_means = [np.mean(region) for region in group]
        group_std_devs = [np.std(region) for region in group]
        means.append(group_means)
        std_devs.append(group_std_devs)

    # Plotting
    #sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(grouped_data))  # the label locations
    width = 0.2  # the width of the bars

    # Labels for the legends as per the user's specification
    legend_labels = ['d:2', 'd:5', 'd:10']
    colors = sns.color_palette("viridis", n_colors=3)
    for i in range(3):  # Three bars per group, adjust for each dimension
        means_plot = [group[i] for group in means]
        std_devs_plot = [group[i] for group in std_devs]
        rects = ax.bar(x - width + i * width, means_plot, width, label=legend_labels[i], 
                    yerr=std_devs_plot, capsize=5, color=colors[i])

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean AUC', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['p:20, FM', 'p:20, LIT_LVM', 'p:30, FM', 'p:30, LIT_LVM', 'p:40, FM', 'p:40, LIT_LVM', 'p:50, FM', 'p:50, LIT_LVM', 'p:60, FM', 'p:60, LIT_LVM'], rotation=35,fontsize=12, fontweight='bold',  ha="right")
    ax.legend()
    ax.set_ylim(0.65, 1)
    fig.tight_layout()
    plt.savefig('LIT_LVM_VS_FMs_V_NOISE_4.pdf', dpi=1000, bbox_inches='tight')
    plt.show()


if Classification_LIT_VS_EN_NOISY:
    # true underlying d is 2
    data = [

[0.9930051116491794, 0.9950316059734476, 0.9897993471582182, 0.9932229128550599, 0.9909863207471348],   #lit_lvm 20 2
[0.9814607350655922, 0.9853101285363753, 0.9744903673835126, 0.985076676400147, 0.977451736831107],#lit_lvm 20 10 
[0.9843197250229884, 0.9865654300524871, 0.9751424091141833, 0.9843597914369605, 0.9800437221713202], #lit_lvm 20 5
[0.9747308654467775, 0.97479354714807, 0.9692364981741304, 0.9709971774969529, 0.9609113745819934], # EN


[0.9571597159715972, 0.9535766294514805, 0.9660346242304941, 0.9534640868062223, 0.9621510673234812], #lit_lvm 30 2
[0.9563796379637964, 0.949143703038059, 0.9619925707056213, 0.9505033288521861, 0.9649384236453202], #lit_lvm 30 5
[0.961000100010001, 0.9522648041689568, 0.966608595831026, 0.9526638819537802, 0.9620771756978654], #lit_lvm 30 10
[0.937969796979698, 0.9201934946055563, 0.9509404187827425, 0.9317578683189836, 0.9279677403760938], #  EN noisy



[0.9398932057891597, 0.9330906527946307, 0.9067281223993342, 0.91404664197778, 0.9269075050758677], # lit_lvm 40 2
[0.9394405521573151, 0.9295385465598232, 0.9031352026118686, 0.9160364327895105, 0.9241763478581241], # lit_lvm 40 5
[0.9399973561823274, 0.9334991249884868, 0.9049196594328147, 0.9137423681313183, 0.9225464637120511], # lit_lvm 40 10
[0.8982931352873549, 0.9066868686868688, 0.8824341907716946, 0.9117992928210501, 0.8882595530382121], # EN



[0.873915722424941, 0.87306231788321, 0.8891611023042065, 0.8797867571721311, 0.8976916900843035], #lit_lvm 50 2
[0.8693305486204469, 0.8680289036315963, 0.8813429728907463, 0.8763167584528688, 0.8921999197109595], #lit_lvm 50 5
[0.8579716407400294, 0.8681707470283808, 0.8781404621223039, 0.8790183145491803, 0.8833801686069852], #lit_lvm 50 10
[0.8270277190960885, 0.8273224572784044, 0.8206725431012022, 0.8505804238677026, 0.8443659829306177], #EN



[0.8341178183229814, 0.8529394030750678, 0.8224808562303905, 0.8311567761346904, 0.8017731649268138], #lit_lvm 60 2
[0.8366491977225673, 0.8254085016581248, 0.8210603504582084, 0.8285761154855641, 0.8032380929446659], #lit_lvm 60 5
[0.8312952898550725, 0.8295769269420159, 0.8216053117699622, 0.8214142820562065, 0.8025598112048933], #lit_lvm 60 10
[0.8013271545031055, 0.7754451981979452, 0.8086033640281897, 0.805784131659836, 0.7908149976558649],  #EN                                                                                      # elasticnet   
    ]
 
    # Assuming 'data' is defined somewhere in your script
    step = 4
    grouped_data = [data[i:i+step] for i in range(0, len(data), step)]

    # Calculate means and standard deviations for each group
    means = []
    std_devs = []
    for group in grouped_data:
        group_means = [np.mean(region) for region in group]
        group_std_devs = [np.std(region) for region in group]
        means.append(group_means)
        std_devs.append(group_std_devs)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(grouped_data))  # the label locations
    width = 0.2  # the width of the bars

    # Calculate the offset to center the bars
    offset = width * (step - 1) / 2

    # Labels for the legends
    legend_labels = ['d:2', 'd:5', 'd:10', 'elastic net']
    colors = sns.color_palette("husl", n_colors=step)
    for i in range(step):
        means_plot = [group[i] for group in means]
        std_devs_plot = [group[i] for group in std_devs]
        rects = ax.bar(x - offset + i * width, means_plot, width, label=legend_labels[i], 
                    yerr=std_devs_plot, capsize=5, color=colors[i])

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean AUC', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['p:20', 'p:30', 'p:40', 'p:50', 'p:60'],rotation=35, fontsize=12, fontweight='bold' , ha="right")
    ax.legend()
    ax.set_ylim(0.4, 1)
    fig.tight_layout()
    plt.savefig('Classification_LIT_VS_EN_NOISY.jpg', dpi=1000, bbox_inches='tight')
    plt.show()


if Classification_LIT_LVM_VS_FMS_SIM_Sparse:
    # true underlying d is 2
    data = [

 [0.886352625799367, 0.8975026142082216, 0.8313774651948926, 0.8948929476325996, 0.8435231127183024], # fm
[0.9577264065628835, 0.9611830926083262, 0.9400142584789888, 0.9704119985109935, 0.9601539379522729], # fm
[0.9890268716491183, 0.9787227958956931, 0.9725324019929829, 0.9804347739071613, 0.9811702327891043],# fm

[0.9908839868225567, 0.9827613867898798, 0.982304667809255, 0.9846335839598999, 0.9919814019742397],  #lit 20
[0.9916591305471223, 0.9855689785179125, 0.984300836777901, 0.9876571428571428, 0.9937779840668377],  #lit 20
[0.9922203023060526, 0.9850997096147984, 0.9819054340155258, 0.9883829573934837, 0.9939260320343791],  #lit 20


[0.8247345919200654, 0.8126096187575003, 0.8233457350527174, 0.8336812384618169, 0.8133005498645378], # fm 30
[0.9098052873452787, 0.8900902636469081, 0.8759908331627595, 0.8954836685690842, 0.848378460699915] ,# fm 30
[0.9428551984756048, 0.9273837188301446, 0.9389305624119538, 0.9412502865692532, 0.9437671331698168], # fm 30


[0.9696361947767049, 0.965658914728682, 0.9707572115384615, 0.9693499087866815, 0.9655219220849736], #lit 30
[0.9708051112071865, 0.9673888208894329, 0.9730528846153845, 0.9693499087866817, 0.9693596752100891],#lit 30
[0.9721421594530112, 0.9699796001631986, 0.9730729166666666, 0.9695351543780379, 0.9693308817782823], #lit 30



[0.8451151323740311, 0.8240865270053594, 0.8194147384794868, 0.8234628064784585, 0.8297918395324506], #fm40
[0.8708325947888301, 0.8556900472134379, 0.8680123468792533, 0.8621767172396133, 0.893210032428619], #fm40
[0.9191293768832884, 0.8892483410676133, 0.907552336509171, 0.9053077587862493, 0.8988940598306016], #fm40


[0.9454994440774102, 0.9145664160401002, 0.9344427331504389, 0.9493151890430247, 0.9390032274282748], # lit 40
[0.9421397379912665, 0.9101914786967418, 0.9142521377805587, 0.9539552632842125, 0.9446593360255598], # lit 40
[0.9427440016758247, 0.9192140350877193, 0.9345991561181434, 0.9504552072833166, 0.9398222481713943], # lit 40



[0.805473280381703, 0.81842104205921, 0.8023083099156966, 0.766257493806191, 0.8142966249045321],    ## fm 50
[0.8409758574416384, 0.8395430495237253, 0.8192131674026495, 0.8492645730186836, 0.8518378589187344], # fm 50
[0.8700589560388634, 0.8551048366685281, 0.8410598153352067, 0.8698396636671366, 0.8626358894359675], # fm 50


[0.8851465438382274, 0.8914863159937888, 0.9050786967418546, 0.9110567263024224, 0.8858481619006284], #lit 50
[0.8917151250649806, 0.8975114842132506, 0.9065784461152883, 0.9121417434215813, 0.8875440034720548], #lit 50
[0.8937461464995103, 0.8930552859730848, 0.9021754385964913, 0.9106066451270676, 0.8920850010448313], #lit 50



[0.8034907745531424, 0.759119207286558, 0.7631755175808039, 0.7683287559759944, 0.8002340818321126], # fm 60
[0.843611858543148, 0.7817715944349916, 0.7992085007503509, 0.8330749669413081, 0.822632736867084], # fm 60
[0.85817076686527, 0.8173956560904815, 0.8444675735424636, 0.8197172210354999, 0.8420792258407037], # fm 60

[0.8463146261772053, 0.8277204515991797, 0.8550895115938059, 0.8434045112781955, 0.8697758251661597], #lit 60
[0.8513918252290347, 0.829763323821335, 0.8535143975893908, 0.8432080200501253, 0.8653663560726759], #lit 60
[0.8504949067845474, 0.8304897675781328, 0.8551660516605166, 0.8447318295739349, 0.8678486940568726],#lit 60
    ]

    grouped_data = [data[i:i+3] for i in range(0, len(data), 3)]

    # Calculate means and standard deviations for each group
    means = []
    std_devs = []
    for group in grouped_data:
        group_means = [np.mean(region) for region in group]
        group_std_devs = [np.std(region) for region in group]
        means.append(group_means)
        std_devs.append(group_std_devs)

    # Plotting
    #sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(grouped_data))  # the label locations
    width = 0.2  # the width of the bars

    # Labels for the legends as per the user's specification
    legend_labels = ['d:2', 'd:5', 'd:10']
    colors = sns.color_palette("viridis", n_colors=3)
    for i in range(3):  # Three bars per group, adjust for each dimension
        means_plot = [group[i] for group in means]
        std_devs_plot = [group[i] for group in std_devs]
        rects = ax.bar(x - width + i * width, means_plot, width, label=legend_labels[i], 
                    yerr=std_devs_plot, capsize=5, color=colors[i])

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean AUC', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['p:20, FM', 'p:20, LIT-LVM', 'p:30, FM', 'p:30, LIT-LVM', 'p:40, FM', 'p:40, LIT-LVM', 'p:50, FM', 'p:50, LIT-LVM', 'p:60, FM', 'p:60, LIT-LVM'], rotation=35,fontsize=12, fontweight='bold',  ha="right")
    ax.legend()
    ax.set_ylim(0.8, 1)
    fig.tight_layout()
    plt.savefig('Classification_LIT_LVM_VS_FMS_SIM_Sparse.pdf', dpi=1000, bbox_inches='tight')
    plt.show()


if Classification_FM_L2_VS_FM_EN:
    # Datasets and mean AUC values
    datasets =                              ['bioresponse', 'clean_1', 'clean_2', 'eye_movement', 'fri_c4_500_100', 'fri_c4_1000_100', 'hill_valley', 'jannis', 'jasmine', 'madeline', 'MiniBooNE', 'nomao', 'pol', 'scene', 'tecator']
    mean_auc = {
        'FM_L2':                            ( 0.795,         0.896,     0.965,     0.585,          0.616,            0.636,             0.655,         0.807,    0.825,     0.610,      0.900,       0.987,   0.954, 0.956,   0.952),
        'FM_EN':                            ( 0.796,         0.894,     0.965,     0.584,          0.696,            0.662,             0.650,         0.807,    0.828,     0.631,      0.900,       0.984,   0.954, 0.966,   0.964),
    }

    auc_std_error = {
        'FM_L2':                            ( 0.006,        0.006,      0.002,     0.007,          0.007,            0.012,             0.03,          0.001,    0.005,     0.008,      0.001,       0.0004,  0.002, 0.002,   0.004),
        'FM_EN':                            ( 0.005,        0.008,      0.006,     0.004,          0.016,            0.007,             0.03,          0.001,    0.003,     0.001,      0.001,       0.002,   0.001, 0.003,   0.003),
    }

    # Set up the figure size
    plt.figure(figsize=(12, 7))

    # Number of datasets
    n_datasets = len(datasets)

    # Bar width
    bar_width = 0.2

    # Set seaborn style
    sns.set(style="white", context='notebook')

    # Use seaborn color palette
    palette = sns.color_palette("husl", 4)

    # Positions of the bars on the x-axis
    r1 = np.arange(n_datasets)
    r2 = [x + bar_width for x in r1]


    # Create bars with specified colors and error bars

    plt.bar(r1, mean_auc['FM_L2'], color=palette[0], width=bar_width, edgecolor='grey', label='FM_L2', yerr=auc_std_error['FM_L2'], capsize=5)
    plt.bar(r2, mean_auc['FM_EN'], color=palette[1], width=bar_width, edgecolor='grey', label='FM_EN', yerr=auc_std_error['FM_EN'], capsize=5)

    # Add xticks on the middle of the group bars
    plt.xticks([r + 1.5 * bar_width for r in range(n_datasets)], datasets, rotation=25, ha='right', fontsize=12, fontweight='bold')

    # Add labels
    plt.ylabel('Mean AUC', fontweight='bold', fontsize=14)

    # Add legend
    plt.legend(loc='upper left', fontsize='large', bbox_to_anchor=(0.35, 1), ncol=1, borderaxespad=0.5, frameon=False)

    # Set y-axis limit
    plt.ylim(0.49, 1)

    # Remove extra white space
    plt.tight_layout(pad=2)

    # Adjust x-axis limits to remove extra space
    plt.xlim(-0.5, n_datasets - 0.5 + 4 * bar_width)

    # Display the chart
    plt.savefig('Classification_FM_L2_VS_FM_EN.pdf', dpi=1000, bbox_inches='tight')
    plt.show()

if classification_Diff_LITLVMvsFM:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Data
    auc_fm = [0.666, 0.665, 0.67, 0.672, 0.669, 0.678]
    auc_fm_err = [0.019, 0.015, 0.011, 0.009, 0.011, 0.009]
    auc_lit = [0.689, 0.697, 0.698, 0.681, 0.686, 0.691]
    auc_lit_err = [0.008, 0.006, 0.005, 0.008, 0.005, 0.009]
    d = [2, 3, 5, 10, 30, 50]

    # Create DataFrame with error values
    data = []
    for i in range(len(d)):
        data.append({'d': d[i], 'Model': 'FM', 'AUC': auc_fm[i], 'Error': auc_fm_err[i]})
        data.append({'d': d[i], 'Model': 'LIT-LVM', 'AUC': auc_lit[i], 'Error': auc_lit_err[i]})

    df = pd.DataFrame(data)

    # Set style and palette
    sns.set_theme(style="white")
    palette = sns.color_palette("Paired", n_colors=2)

    # Create plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='d', y='AUC', hue='Model', data=df, palette=palette)

    # Add error bars
    for patch, err in zip(ax.patches, df['Error']):
        x = patch.get_x() + patch.get_width()/2
        y = patch.get_height()
        ax.errorbar(x, y, yerr=err, 
                fmt='none',        # No connecting line
                color='black',     # Black error bars
                capsize=5,         # Cap size for error bars
                linewidth=1)       # Line width

    # Customize plot
    plt.title('Dataset: fri_c4_1000_100', fontsize=14, pad=20)
    plt.xlabel('Low-rank Dimension (d)', fontsize=12)
    plt.ylabel('Mean AUC', fontsize=12)
    plt.ylim(0.5, 0.75)
    plt.legend(title='Model Type', loc='upper right')

    # # Add value labels
    # for p in ax.patches:
    #     ax.annotate(f"{p.get_height():.3f}", 
    #                 (p.get_x() + p.get_width() / 2., p.get_height()),
    #                 ha='center', va='center', 
    #                 xytext=(0, 9), 
    #                 textcoords='offset points')

    # Save plot
    plt.savefig('classification_Diff_LITLVMvsFM.pdf', dpi=1000, bbox_inches='tight')
    plt.close()


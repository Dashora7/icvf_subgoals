import re
from ast import literal_eval
import numpy as np

ICVF_SUBGOAL_STATES = """[[ 1.12917116e-02  6.17409013e-02  7.43387878e-01  9.93142247e-01
   8.67436305e-02 -7.51166791e-02 -2.23949309e-02  4.85203054e-04
   1.16876978e-02 -2.42947452e-02 -2.08744295e-02  5.08697964e-02
  -5.19658700e-02  7.48153031e-02 -8.71660337e-02  1.72014356e-01
   3.43807368e-03 -1.12877987e-01 -1.26453578e-01  9.16344970e-02
  -6.96259215e-02 -2.85325404e-02  1.30809555e-02  1.34038299e-01
  -5.51618710e-02  5.77314161e-02 -1.70989484e-02  1.34212952e-02
   4.89007607e-02]
 [ 3.94189024e+00 -5.63195348e-02  5.18152475e-01  9.81517732e-01
   7.31104910e-02 -1.29784001e-02 -1.76378548e-01  5.42541504e-01
   4.51413780e-01  1.10112548e-01 -8.91646326e-01 -3.31316561e-01
  -4.99650300e-01  1.97317421e-01  5.22593200e-01  2.36177635e+00
  -8.03011537e-01 -6.55985475e-01  1.37810791e+00  3.02515578e+00
   4.75105762e-01  6.29127800e-01  9.43469286e-01  5.34654093e+00
  -2.79193950e+00 -2.55589485e+00 -3.86983395e-01 -4.41680908e+00
   8.04320816e-03]
 [ 8.07519341e+00  1.10715342e+00  6.28383458e-01  9.34894264e-01
  -8.21766406e-02  9.00631770e-02 -3.33329231e-01 -5.25618613e-01
   5.22677660e-01  5.32091677e-01 -1.10783482e+00 -6.33413568e-02
  -6.39803648e-01 -5.24450660e-01  1.27225590e+00  2.30396318e+00
   1.22383825e-01  2.64690489e-01 -2.16460299e+00  4.43171084e-01
   6.72292531e-01  2.57917913e-03  2.21927434e-01 -7.45398343e-01
   1.82347393e+00  5.20227051e+00 -2.32272744e+00 -5.10603046e+00
   8.62676919e-01]
 [ 8.49890137e+00  3.99662638e+00  7.39395678e-01  9.56240356e-01
  -6.70060664e-02  7.28895888e-02 -2.75320977e-01 -5.41355848e-01
   1.16652107e+00 -4.22166318e-01 -6.32214189e-01  5.59842765e-01
  -8.98180783e-01  4.40076776e-02  6.42969608e-01 -7.41331160e-01
   2.09534287e+00 -1.97746381e-01 -1.28952086e+00 -1.83207345e+00
  -1.48038697e+00  1.16993690e+00 -2.01586032e+00  2.03655267e+00
  -2.16076326e+00 -3.55669230e-01  2.01862025e+00  5.16002607e+00
   2.37957144e+00]
 [ 8.74087906e+00  8.59321976e+00  6.18826866e-01  9.87437844e-01
   5.77838905e-02  1.27593681e-01 -7.31255710e-02 -6.16913199e-01
   4.99888688e-01  3.10068071e-01 -5.19211233e-01  5.75850368e-01
  -1.24035120e+00 -5.55252194e-01  6.39331579e-01 -1.24826646e+00
   1.45038807e+00  6.68057024e-01  2.34950140e-01 -2.55282491e-01
   5.69679618e-01 -2.34041631e-01  3.49907815e-01  5.12701225e+00
  -5.30247949e-02 -5.83467066e-01  1.75884712e+00  1.75139368e-01
   2.31887197e+00]
 [ 4.06380033e+00  7.85722399e+00  5.59333026e-01  8.94754410e-01
  -3.04515194e-02 -4.64066379e-02  4.43095535e-01 -7.76000321e-02
   9.51702893e-01 -5.68457305e-01 -5.21744609e-01  3.02467197e-01
  -1.10938764e+00 -3.79132003e-01  5.08899748e-01 -1.75310767e+00
  -8.35615098e-02 -4.21993211e-02 -2.10560179e+00 -4.43218756e+00
  -5.03556490e-01  4.82720375e+00  4.53859234e+00 -4.58593130e+00
   7.05642160e-03 -3.22643042e+00  2.75514674e+00  2.04444066e-01
   1.77260146e-01]
 [ 1.12927032e+00  8.83936405e+00  4.73164529e-01  9.99929368e-01
   1.11397747e-02  4.04335558e-03 -9.11273120e-04 -5.55445433e-01
   5.08679748e-01  2.75538504e-01 -7.95863211e-01  5.45327246e-01
  -6.36435747e-01 -1.48495644e-01  6.62715018e-01 -1.72403884e+00
   1.62726545e+00 -3.01542789e-01  2.40586847e-02 -1.40279162e+00
  -3.74620914e-01  2.25138187e+00 -6.44434988e-02  5.43837023e+00
  -3.49459243e+00 -3.68613958e-01  2.16565967e+00 -4.05809450e+00
   2.73081732e+00]]"""

ICVF_SUBGOAL_STATES = re.sub(r"([^[])\s+([^]])", r"\1, \2", ICVF_SUBGOAL_STATES)
SUBGOALS_small = np.array(literal_eval(ICVF_SUBGOAL_STATES))

SUBGOALS_medium = np.array([[
    4.34921408e+00,  8.28593850e-01,  4.67559367e-01,
    9.80559111e-01,  5.28712459e-02, -6.38997275e-03,
  -1.88858807e-01,  5.26024044e-01,  4.64006484e-01,
  -2.19395533e-01, -7.53936529e-01,  4.89629120e-01,
  -5.21167397e-01, -6.27141306e-03,  5.21552861e-01,
    5.20836532e-01,  1.42515314e+00, -9.78123248e-01,
  -6.64193407e-02, -1.34498668e+00, -1.90047905e-01,
  -1.73436000e-03,  2.73321778e-01,  3.94622254e+00,
  -2.79979038e+00,  1.69115114e+00, -6.98856136e-04,
  -3.57785392e+00, -1.30969216e-03],
  [ 4.88865185e+00,  5.06735563e+00,  6.46498501e-01,
    9.70108390e-01,  5.93310557e-02, -9.23334733e-02,
  -2.16434941e-01, -1.65494546e-01,  8.42121840e-01,
  -2.30903104e-01, -6.89885676e-01, -1.29139451e-02,
  -5.13420641e-01,  6.00876629e-01,  6.74942136e-01,
  -3.59220058e-01,  1.63823378e+00, -8.36625099e-01,
  -3.12775820e-01, -6.75470114e-01, -3.28391969e-01,
    4.15058708e+00, -4.11329079e+00,  3.70321941e+00,
  -2.98647833e+00, -5.83704901e+00, -1.52093291e-01,
    3.48017931e-01,  1.56435537e+00],
  [ 4.96862221e+00,  8.52211952e+00,  4.97728914e-01,
    9.97028053e-01, -1.81156036e-03, -4.80319113e-02,
  -6.02053404e-02, -9.27926600e-02,  5.87695658e-01,
    3.19313854e-01, -6.26507759e-01,  5.15995957e-02,
  -5.22254348e-01,  1.87399253e-01,  9.95997369e-01,
    2.20436692e-01,  1.64544463e+00, -1.20336974e+00,
  -3.42484426e+00, -5.83206654e-01,  7.79260933e-01,
  -5.84879780e+00,  1.31250930e+00, -3.50233912e+00,
    3.66174865e+00,  2.28064704e+00, -5.46776515e-04,
    1.79707408e+00,  4.09870577e+00],
  [ 8.79615879e+00,  8.01733494e+00,  7.52435684e-01,
    9.68996882e-01,  5.45198880e-02, -1.47056416e-01,
  -1.90911070e-01,  2.75752813e-01,  5.48656166e-01,
    5.63656032e-01, -8.40887964e-01, -2.33406901e-01,
  -5.05617321e-01,  5.34249842e-01,  5.21138310e-01,
    1.99662817e+00, -3.83739829e-01,  2.66268939e-01,
    1.58062208e+00,  2.94783306e+00, -6.30375624e-01,
  -3.30408764e+00, -5.54435396e+00,  5.59084272e+00,
  -3.83314300e+00, -5.20469856e+00, -2.93648511e-01,
  -1.68727130e-01,  4.66447137e-03],
  [ 1.30226774e+01,  8.10419464e+00,  5.25250852e-01,
    9.52564240e-01,  8.97736102e-02,  5.44357672e-03,
  -2.90744722e-01,  4.05095577e-01,  6.76973343e-01,
  -5.72162390e-01, -6.16939425e-01, -2.47236773e-01,
  -6.60181522e-01, -5.45313239e-01,  5.05445838e-01,
  -1.05606711e+00,  7.55831838e-01,  6.68865219e-02,
  -6.46841004e-02, -1.35042584e+00,  8.47012281e-01,
  -2.57283735e+00,  2.99659109e+00, -2.01637477e-01,
    5.69414854e+00,  4.59092331e+00, -2.74457741e+00,
    1.09992313e+00,  2.80087322e-01],
  [ 1.33295774e+01,  1.21702032e+01,  5.00173211e-01,
    9.97029603e-01,  4.42827828e-02,  3.00958697e-02,
  -5.53641282e-02,  4.42778885e-01,  6.53764963e-01,
    4.16334420e-01, -8.61852229e-01, -4.03451145e-01,
  -6.15976691e-01, -4.43219155e-01,  5.43800592e-01,
    1.01470578e+00,  1.59565818e+00, -9.50412512e-01,
  -2.18252921e+00,  1.37749732e-01, -1.95954323e-01,
  -2.75561357e+00,  2.59856296e+00, -2.09512305e+00,
    3.11846018e+00,  3.13086581e+00, -1.87142634e+00,
    2.38206482e+00,  5.63651979e-01],
  [ 1.64436607e+01,  1.28869743e+01,  9.28888738e-01,
    9.75238562e-01, -5.76154627e-02, -1.12881176e-01,
    1.81240410e-01,  5.88704526e-01,  1.22267997e+00,
  -5.53856611e-01, -5.36561310e-01,  5.32197297e-01,
  -5.19267976e-01,  5.61155736e-01,  5.05044639e-01,
    1.48724580e+00,  1.66686028e-01, -2.89809644e-01,
    1.48055267e+00, -8.97938967e-01,  1.17511690e+00,
  -8.50104928e-01, -7.45917857e-01,  5.35095930e-01,
  -2.93896824e-01, -1.47742376e-01,  2.77041197e+00,
  -1.89114904e+00,  3.00137192e-01],
  [ 2.02981987e+01,  1.27838116e+01,  3.61169785e-01,
    9.38365579e-01, -1.88278556e-02,  2.16119755e-02,
    3.44453871e-01,  5.27680874e-01,  4.49510068e-01,
    6.89656660e-02, -6.71775341e-01,  4.29788560e-01,
  -4.55632269e-01, -4.23324049e-01,  5.39066851e-01,
    7.42822230e-01,  7.78174639e-01,  3.10926465e-04,
  -7.03882158e-01,  1.50152373e+00,  2.76502669e-01,
    6.61933348e-02,  9.83421504e-01, -4.03955787e-01,
  -2.38385963e+00,  2.03501916e+00, -8.61769319e-01,
  -1.38258100e+00,  6.70231953e-02],
  [ 2.02894211e+01,  1.70118904e+01,  5.28708458e-01,
    6.22345328e-01,  8.86064321e-02,  1.71073806e-02,
  -7.77523339e-01,  7.12857246e-02,  7.77759016e-01,
    5.49861908e-01, -4.95218247e-01,  4.13996503e-02,
  -1.13755870e+00, -3.73951048e-01,  5.20780325e-01,
    9.31284010e-01,  1.83243072e+00,  1.55734444e+00,
    3.33252490e-01,  1.22984159e+00, -2.01862410e-01,
  -4.60623598e+00,  2.38165903e+00, -3.06197792e-01,
  -4.79468077e-01,  5.22811651e+00, -3.51781702e+00,
    2.96379495e+00,  6.06920198e-03],
  [ 2.04760895e+01,  2.06761723e+01,  6.27628148e-01,
    9.83829439e-01,  8.62841755e-02, -4.84127626e-02,
  -1.49301305e-01,  1.88552320e-01,  8.75363827e-01,
    2.64983654e-01, -1.26207042e+00, -3.43144178e-01,
  -6.14107847e-01, -5.87922812e-01,  7.34830439e-01,
    1.51591837e-01,  1.68437719e+00,  6.83475137e-02,
    6.58433735e-01,  1.00202227e+00,  6.78965330e-01,
    2.30785346e+00, -2.20885134e+00,  3.19339991e+00,
    5.18691480e-01, -6.05844355e+00, -2.26691580e+00,
    5.75765193e-01, -2.52057481e+00]])


# ICVF_SUBGOAL_STATES_2 = re.sub(r"([^[])\s+([^]])", r"\1, \2", ICVF_SUBGOAL_STATES_2)
# SUBGOALS_medium = np.array(literal_eval(ICVF_SUBGOAL_STATES_2))

SUBGOALS_hard = np.array([[
  8.54381180e+00,  8.25611234e-01,  7.18255937e-01,
  9.99086261e-01,  2.09736712e-02, -3.39104868e-02,
  -1.53887803e-02, -4.93415929e-02,  1.60989165e-02,
  -4.51364741e-02,  7.09443167e-02, -4.97511737e-02,
  -4.15691398e-02,  4.14914228e-02, -1.94049049e-02,
    1.14944782e-02,  9.24729034e-02,  3.60884964e-02,
  -1.48510635e-01, -1.57301858e-01, -1.33682042e-01,
  -9.52520669e-02, -1.47449285e-01, -5.50922528e-02,
    4.14212719e-02, -1.54679447e-01, -1.36935189e-02,
    2.90661994e-02,  1.79918304e-01],
  [ 1.28140488e+01,  1.12306440e+00,  5.30647159e-01,
    9.67743158e-01,  5.06735891e-02,  1.33359320e-02,
  -2.46429577e-01, -5.62571347e-01,  8.18728805e-01,
  -5.23235500e-01, -5.21244228e-01,  6.18212938e-01,
  -6.25365317e-01,  7.40547199e-03,  5.21365523e-01,
  -8.33445132e-01,  2.11190724e+00,  1.08169007e+00,
    7.19046712e-01, -8.39627981e-02,  6.27189875e-02,
  -2.66783404e+00,  2.92297530e+00, -8.87893513e-03,
  -2.64297915e-03, -7.50287533e-01,  7.43707776e-01,
    1.51909792e+00,  2.02625524e-03],
  [ 1.28970299e+01,  4.26844549e+00,  6.81207240e-01,
    9.80287910e-01,  1.16419598e-01, -1.20408118e-01,
  -1.04804508e-01,  3.49095166e-01,  5.06917357e-01,
    6.18355460e-02, -1.25112009e+00, -5.36433339e-01,
  -5.21224737e-01, -5.62616289e-01,  9.36561167e-01,
    7.68321037e-01,  7.91884243e-01,  4.24686283e-01,
    1.47024536e+00, -4.38005924e-01, -2.74804503e-01,
    6.86881876e+00,  2.46468544e-01,  4.55701262e-01,
    8.02711695e-02, -5.75210524e+00, -1.10992072e-02,
    2.02440664e-01, -3.02432799e+00],
  [ 1.39814939e+01,  8.90574265e+00,  7.80740559e-01,
    8.96681368e-01, -1.21652551e-01, -1.29545912e-01,
  -4.05439258e-01, -5.68372943e-02,  5.84303498e-01,
    3.58638585e-01, -7.40753651e-01, -9.34931412e-02,
  -1.26544595e+00, -4.97996271e-01,  1.15604579e+00,
    8.16403806e-01,  6.16170883e-01, -1.91960201e-01,
  -9.96520340e-01, -1.07809329e+00, -4.75331932e-01,
    4.00124884e+00,  1.25001490e+00, -7.82633007e-01,
    2.43399930e+00, -6.16274297e-01, -4.72773552e+00,
    1.39785564e+00, -2.12734056e+00],
  [ 1.79541950e+01,  8.07465172e+00,  5.92106819e-01,
    9.64578271e-01, -3.88497189e-02, -1.01701915e-01,
    2.40283623e-01,  3.50504249e-01,  1.17317057e+00,
  -2.91544706e-01, -4.85352516e-01, -1.78028956e-01,
  -1.23517251e+00, -1.19442821e-01,  6.68725431e-01,
    1.83363557e+00, -1.47585263e-02,  9.74820137e-01,
    6.58880591e-01, -1.90233135e+00, -4.80673015e-01,
    5.86260414e+00,  4.98552799e+00, -6.26529026e+00,
    4.46776360e-01, -4.45262003e+00,  4.34190273e-01,
    4.95665503e+00, -5.52876377e+00],
  [ 2.10320339e+01,  9.48960209e+00,  4.83434051e-01,
    9.97729897e-01,  5.83167970e-02,  1.47320256e-02,
    3.02848797e-02,  5.25047719e-01,  4.79710340e-01,
    3.61139402e-02, -1.12162817e+00,  3.17831755e-01,
  -5.21089077e-01, -4.14800733e-01,  5.15349627e-01,
    1.28666604e+00,  2.16494083e+00,  1.26282668e+00,
    6.52351510e-03,  8.73577476e-01,  9.17506814e-01,
  -7.19209015e-03,  6.35259807e-01,  4.86883211e+00,
  -5.01720428e+00, -3.67987657e+00, -3.45730921e-03,
  -5.71718931e+00,  2.59407878e-01],
  [ 2.09146252e+01,  1.22391911e+01,  4.49856550e-01,
    9.47111666e-01,  1.01741636e-03,  1.79798175e-02,
  -3.20398390e-01,  7.96622336e-02,  6.55090868e-01,
  -5.12385428e-01, -5.09561896e-01,  2.88246840e-01,
  -7.54727423e-01,  4.51799482e-02,  5.21798790e-01,
  -6.88413203e-01,  2.61223674e+00,  1.05258751e+00,
    8.19767058e-01, -4.60296609e-02, -5.43348372e-01,
  -3.51052976e+00,  2.52682567e+00, -1.22657156e+00,
  -2.24192396e-01,  6.22863531e+00, -1.85532331e-01,
    3.09109759e+00,  3.83088924e-03],
  [ 2.09922218e+01,  1.59021502e+01,  4.75575507e-01,
    9.97150242e-01,  7.36668929e-02, -1.37078390e-02,
  -8.75553396e-03, -6.17231667e-01,  9.82191324e-01,
  -2.30773106e-01, -4.48535085e-01,  3.48840743e-01,
  -5.06464422e-01,  4.27926570e-01,  5.20910323e-01,
  -1.69676587e-01,  1.19361627e+00,  5.74503839e-01,
  -4.44789082e-01, -5.27810454e-01, -1.20882213e+00,
  -1.52043843e+00,  4.05133629e+00, -4.22325563e+00,
  -6.40872896e-01,  4.43043232e+00,  5.90025008e-01,
    1.56046641e+00,  6.74800202e-03],
  [ 2.41053963e+01,  1.60914707e+01,  5.04960597e-01,
    9.92170274e-01,  2.44753882e-02, -1.22470535e-01,
    1.56321767e-04, -4.46077764e-01,  1.12403035e+00,
  -3.74000132e-01, -5.18137097e-01,  3.71129096e-01,
  -4.87446755e-01,  4.63313550e-01,  5.22002876e-01,
    3.96150500e-01,  7.02370882e-01, -7.55715728e-01,
  -1.38652241e+00, -4.17756438e-01,  1.71489108e+00,
  -2.02232313e+00, -3.11614203e+00,  3.01610112e+00,
  -7.60808960e-02, -3.07213187e+00, -3.80770057e-01,
  -2.34509754e+00,  4.56531160e-03],
  [ 2.85205956e+01,  1.58661909e+01,  5.01827121e-01,
    9.91152585e-01,  2.68533304e-02,  8.99676755e-02,
    9.38148871e-02,  2.95947194e-01,  5.21248400e-01,
    6.37086689e-01, -1.30329394e+00,  1.85512409e-01,
  -5.30933678e-01, -5.30201077e-01,  5.22242188e-01,
    1.30247796e+00,  6.98424280e-01, -2.52566457e-01,
  -2.91106671e-01,  4.89005804e-01, -4.93716508e-01,
    3.70986080e+00, -1.01181481e-03,  1.17593862e-01,
    3.64766538e-01, -2.09135675e+00, -1.91651374e-01,
    8.10832307e-02,  2.13611359e-03],
  [ 2.82985573e+01,  2.04171238e+01,  6.10670328e-01,
    9.48024988e-01,  3.13193291e-01, -5.60900234e-02,
  -3.53387184e-03,  5.38220942e-01,  5.21034777e-01,
    5.16906619e-01, -1.19582140e+00, -5.97773314e-01,
  -5.22203326e-01,  1.92682996e-01,  5.20905435e-01,
  -3.59715223e-01,  8.53078365e-01,  6.88777208e-01,
    1.75209427e+00, -9.49756801e-01, -4.51708436e-01,
  -2.38041490e-01,  2.22238209e-02, -1.99335289e+00,
    2.31953168e+00,  1.50626504e+00, -2.64816941e-03,
  -2.94210768e+00,  5.88501757e-03],
  [ 2.84376965e+01,  2.39416142e+01,  4.72394139e-01,
    9.22667682e-01,  1.14491083e-01, -4.60409746e-02,
  -3.65316778e-01,  6.53531790e-01,  4.93716449e-01,
  -5.44122994e-01, -1.01084793e+00, -6.37427926e-01,
  -5.22764981e-01,  2.74805516e-01,  4.65047181e-01,
    8.66039217e-01,  1.68213046e+00, -9.46706772e-01,
    9.60231796e-02, -1.18345785e+00,  5.56976318e-01,
    1.32193089e+00,  4.57444459e-01, -2.40429044e+00,
    3.94488883e+00,  3.27634335e-01, -5.91219217e-03,
  -4.13711405e+00,  2.16092601e-01],
  [ 3.14563923e+01,  2.42540798e+01,  4.57793087e-01,
    9.34116542e-01,  3.26864608e-02, -3.51070575e-02,
  -3.53730589e-01,  4.94554490e-01,  4.68409508e-01,
    3.53935151e-03, -9.59431469e-01,  1.68310031e-01,
  -4.81840521e-01, -4.27550217e-03,  5.21680355e-01,
    2.72739029e+00, -5.96117735e-01, -4.15369004e-01,
    1.07335663e+00,  4.29510355e+00, -6.37525320e-01,
  -6.05515778e-01, -2.40943599e+00,  5.29783773e+00,
  -4.04027987e+00, -3.88461828e+00, -6.77895844e-01,
  -5.37068796e+00,  1.53756589e-02]])



SUBGOALS = {'small': SUBGOALS_small, 'medium': SUBGOALS_medium, 'large': SUBGOALS_hard}

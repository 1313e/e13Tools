
from matplotlib.colors import ListedColormap

cm_type = "linear"

cm_data = [[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
           [2.33171379e-04, 2.30277145e-04, 2.74527507e-04],
           [8.04618956e-04, 7.98357412e-04, 9.78696655e-04],
           [1.65943451e-03, 1.65368583e-03, 2.07937053e-03],
           [2.77258813e-03, 2.77420763e-03, 3.57020156e-03],
           [4.12796982e-03, 4.14626223e-03, 5.45173197e-03],
           [5.71384990e-03, 5.76031172e-03, 7.72779252e-03],
           [7.52106657e-03, 7.60925143e-03, 1.04042114e-02],
           [9.54214633e-03, 9.68757701e-03, 1.34880607e-02],
           [1.17708304e-02, 1.19909190e-02, 1.69870780e-02],
           [1.42017007e-02, 1.45157369e-02, 2.09098047e-02],
           [1.68299580e-02, 1.72591232e-02, 2.52655309e-02],
           [1.96513259e-02, 2.02186812e-02, 3.00639016e-02],
           [2.26619646e-02, 2.33924335e-02, 3.53147318e-02],
           [2.58583238e-02, 2.67787353e-02, 4.10162996e-02],
           [2.92371099e-02, 3.03762265e-02, 4.68033596e-02],
           [3.27951699e-02, 3.41837741e-02, 5.25675026e-02],
           [3.65296325e-02, 3.82004809e-02, 5.83133115e-02],
           [4.04377416e-02, 4.23654461e-02, 6.40449173e-02],
           [4.43371164e-02, 4.64819540e-02, 6.97660223e-02],
           [4.81816522e-02, 5.05544107e-02, 7.54799015e-02],
           [5.19746336e-02, 5.45860426e-02, 8.11894744e-02],
           [5.57187871e-02, 5.85797283e-02, 8.68974638e-02],
           [5.94165314e-02, 6.25380542e-02, 9.26063336e-02],
           [6.30700920e-02, 6.64633724e-02, 9.83180132e-02],
           [6.66814153e-02, 7.03578077e-02, 1.04034465e-01],
           [7.02522464e-02, 7.42232969e-02, 1.09757491e-01],
           [7.37841531e-02, 7.80616126e-02, 1.15488751e-01],
           [7.72785466e-02, 8.18743834e-02, 1.21229783e-01],
           [8.07366986e-02, 8.56631105e-02, 1.26982017e-01],
           [8.41597562e-02, 8.94291826e-02, 1.32746786e-01],
           [8.75487542e-02, 9.31738883e-02, 1.38525338e-01],
           [9.09046262e-02, 9.68984267e-02, 1.44318844e-01],
           [9.42282138e-02, 1.00603917e-01, 1.50128408e-01],
           [9.75202746e-02, 1.04291407e-01, 1.55955069e-01],
           [1.00781489e-01, 1.07961878e-01, 1.61799811e-01],
           [1.04012469e-01, 1.11616256e-01, 1.67663567e-01],
           [1.07213757e-01, 1.15255412e-01, 1.73547224e-01],
           [1.10385840e-01, 1.18880169e-01, 1.79451622e-01],
           [1.13529145e-01, 1.22491309e-01, 1.85377565e-01],
           [1.16644047e-01, 1.26089572e-01, 1.91325815e-01],
           [1.19730755e-01, 1.29675641e-01, 1.97297537e-01],
           [1.22789631e-01, 1.33250205e-01, 2.03293117e-01],
           [1.25820921e-01, 1.36813908e-01, 2.09313169e-01],
           [1.28824820e-01, 1.40367367e-01, 2.15358330e-01],
           [1.31801479e-01, 1.43911173e-01, 2.21429212e-01],
           [1.34750946e-01, 1.47445881e-01, 2.27526627e-01],
           [1.37673166e-01, 1.50972017e-01, 2.33651583e-01],
           [1.40568349e-01, 1.54490141e-01, 2.39804013e-01],
           [1.43436501e-01, 1.58000766e-01, 2.45984427e-01],
           [1.46277439e-01, 1.61504360e-01, 2.52193832e-01],
           [1.49091036e-01, 1.65001400e-01, 2.58432884e-01],
           [1.51877389e-01, 1.68492386e-01, 2.64701390e-01],
           [1.54636193e-01, 1.71977750e-01, 2.71000337e-01],
           [1.57367168e-01, 1.75457924e-01, 2.77330492e-01],
           [1.60070392e-01, 1.78933391e-01, 2.83691361e-01],
           [1.62745192e-01, 1.82404518e-01, 2.90084693e-01],
           [1.65391624e-01, 1.85871779e-01, 2.96509830e-01],
           [1.68009165e-01, 1.89335562e-01, 3.02967779e-01],
           [1.70597506e-01, 1.92796288e-01, 3.09458769e-01],
           [1.73156235e-01, 1.96254363e-01, 3.15983210e-01],
           [1.75684919e-01, 1.99710195e-01, 3.22541447e-01],
           [1.78182987e-01, 2.03164175e-01, 3.29134093e-01],
           [1.80650173e-01, 2.06616739e-01, 3.35760757e-01],
           [1.83085426e-01, 2.10068228e-01, 3.42423136e-01],
           [1.85488654e-01, 2.13519110e-01, 3.49120099e-01],
           [1.87858967e-01, 2.16969762e-01, 3.55852595e-01],
           [1.90195513e-01, 2.20420576e-01, 3.62621301e-01],
           [1.92497853e-01, 2.23871999e-01, 3.69425641e-01],
           [1.94765156e-01, 2.27324445e-01, 3.76265946e-01],
           [1.96996298e-01, 2.30778310e-01, 3.83143124e-01],
           [1.99190486e-01, 2.34234037e-01, 3.90057046e-01],
           [2.01346869e-01, 2.37692078e-01, 3.97007588e-01],
           [2.03464399e-01, 2.41152880e-01, 4.03994929e-01],
           [2.05541967e-01, 2.44616901e-01, 4.11019207e-01],
           [2.07578404e-01, 2.48084613e-01, 4.18080507e-01],
           [2.09572473e-01, 2.51556503e-01, 4.25178858e-01],
           [2.11522872e-01, 2.55033071e-01, 4.32314228e-01],
           [2.13428228e-01, 2.58514839e-01, 4.39486516e-01],
           [2.15287095e-01, 2.62002344e-01, 4.46695545e-01],
           [2.17097952e-01, 2.65496146e-01, 4.53941054e-01],
           [2.18859201e-01, 2.68996828e-01, 4.61222691e-01],
           [2.20568556e-01, 2.72504964e-01, 4.68541227e-01],
           [2.22224688e-01, 2.76021217e-01, 4.75895158e-01],
           [2.23825760e-01, 2.79546253e-01, 4.83283750e-01],
           [2.25368961e-01, 2.83080732e-01, 4.90707812e-01],
           [2.26852716e-01, 2.86625410e-01, 4.98165397e-01],
           [2.28274203e-01, 2.90181035e-01, 5.05656556e-01],
           [2.29631280e-01, 2.93748426e-01, 5.13179675e-01],
           [2.30920575e-01, 2.97328417e-01, 5.20734907e-01],
           [2.32139897e-01, 3.00921925e-01, 5.28319852e-01],
           [2.33286007e-01, 3.04529898e-01, 5.35933474e-01],
           [2.34355049e-01, 3.08153341e-01, 5.43575168e-01],
           [2.35343955e-01, 3.11793336e-01, 5.51242462e-01],
           [2.36248975e-01, 3.15451027e-01, 5.58933391e-01],
           [2.37066091e-01, 3.19127633e-01, 5.66645747e-01],
           [2.37791074e-01, 3.22824453e-01, 5.74376926e-01],
           [2.38418694e-01, 3.26542887e-01, 5.82124964e-01],
           [2.38944707e-01, 3.30284414e-01, 5.89885687e-01],
           [2.39364262e-01, 3.34050616e-01, 5.97654955e-01],
           [2.39670357e-01, 3.37843235e-01, 6.05430392e-01],
           [2.39858557e-01, 3.41664055e-01, 6.13205367e-01],
           [2.39921821e-01, 3.45515059e-01, 6.20975383e-01],
           [2.39853218e-01, 3.49398358e-01, 6.28734538e-01],
           [2.39646145e-01, 3.53316179e-01, 6.36475324e-01],
           [2.39293189e-01, 3.57270922e-01, 6.44189794e-01],
           [2.38786055e-01, 3.61265184e-01, 6.51869412e-01],
           [2.38117129e-01, 3.65301666e-01, 6.59503461e-01],
           [2.37278621e-01, 3.69383220e-01, 6.67079890e-01],
           [2.36262155e-01, 3.73512884e-01, 6.74585545e-01],
           [2.35059583e-01, 3.77693824e-01, 6.82005403e-01],
           [2.33664382e-01, 3.81929190e-01, 6.89321651e-01],
           [2.32069800e-01, 3.86222262e-01, 6.96515171e-01],
           [2.30272156e-01, 3.90576102e-01, 7.03563550e-01],
           [2.28269532e-01, 3.94993631e-01, 7.10442430e-01],
           [2.26063977e-01, 3.99477335e-01, 7.17124823e-01],
           [2.23662020e-01, 4.04029120e-01, 7.23581758e-01],
           [2.21076959e-01, 4.08649947e-01, 7.29782417e-01],
           [2.18329680e-01, 4.13339603e-01, 7.35695345e-01],
           [2.15449884e-01, 4.18096420e-01, 7.41289724e-01],
           [2.12477606e-01, 4.22916942e-01, 7.46536881e-01],
           [2.09463074e-01, 4.27795815e-01, 7.51412347e-01],
           [2.06465491e-01, 4.32725854e-01, 7.55897846e-01],
           [2.03551230e-01, 4.37698244e-01, 7.59982868e-01],
           [2.00790629e-01, 4.42702971e-01, 7.63665648e-01],
           [1.98254295e-01, 4.47729383e-01, 7.66953291e-01],
           [1.96009139e-01, 4.52766823e-01, 7.69860989e-01],
           [1.94114951e-01, 4.57805201e-01, 7.72410527e-01],
           [1.92621965e-01, 4.62835416e-01, 7.74628409e-01],
           [1.91569348e-01, 4.67849635e-01, 7.76543907e-01],
           [1.90985240e-01, 4.72841292e-01, 7.78187567e-01],
           [1.90885698e-01, 4.77805369e-01, 7.79589234e-01],
           [1.91277410e-01, 4.82737902e-01, 7.80777858e-01],
           [1.92157514e-01, 4.87636156e-01, 7.81780144e-01],
           [1.93516056e-01, 4.92498211e-01, 7.82620837e-01],
           [1.95336929e-01, 4.97322934e-01, 7.83322277e-01],
           [1.97599340e-01, 5.02109824e-01, 7.83904389e-01],
           [2.00279192e-01, 5.06858881e-01, 7.84384786e-01],
           [2.03350478e-01, 5.11570398e-01, 7.84779328e-01],
           [2.06785951e-01, 5.16244993e-01, 7.85101857e-01],
           [2.10558133e-01, 5.20883449e-01, 7.85364775e-01],
           [2.14639780e-01, 5.25486773e-01, 7.85578633e-01],
           [2.19004580e-01, 5.30056012e-01, 7.85753024e-01],
           [2.23627381e-01, 5.34592295e-01, 7.85896348e-01],
           [2.28484449e-01, 5.39096805e-01, 7.86015955e-01],
           [2.33553633e-01, 5.43570750e-01, 7.86118277e-01],
           [2.38814372e-01, 5.48015301e-01, 7.86209326e-01],
           [2.44247775e-01, 5.52431669e-01, 7.86294111e-01],
           [2.49836618e-01, 5.56821063e-01, 7.86376932e-01],
           [2.55565073e-01, 5.61184601e-01, 7.86462190e-01],
           [2.61418931e-01, 5.65523442e-01, 7.86553220e-01],
           [2.67385341e-01, 5.69838701e-01, 7.86653101e-01],
           [2.73452573e-01, 5.74131425e-01, 7.86764969e-01],
           [2.79610172e-01, 5.78402659e-01, 7.86891329e-01],
           [2.85848805e-01, 5.82653420e-01, 7.87034320e-01],
           [2.92160043e-01, 5.86884681e-01, 7.87196017e-01],
           [2.98536346e-01, 5.91097378e-01, 7.87378267e-01],
           [3.04970976e-01, 5.95292419e-01, 7.87582720e-01],
           [3.11457922e-01, 5.99470679e-01, 7.87810850e-01],
           [3.17991780e-01, 6.03632999e-01, 7.88064059e-01],
           [3.24567662e-01, 6.07780186e-01, 7.88343719e-01],
           [3.31181434e-01, 6.11913034e-01, 7.88650686e-01],
           [3.37829318e-01, 6.16032296e-01, 7.88985923e-01],
           [3.44507793e-01, 6.20138696e-01, 7.89350557e-01],
           [3.51213857e-01, 6.24232937e-01, 7.89745396e-01],
           [3.57945018e-01, 6.28315697e-01, 7.90170911e-01],
           [3.64698543e-01, 6.32387627e-01, 7.90628337e-01],
           [3.71472702e-01, 6.36449356e-01, 7.91117743e-01],
           [3.78265294e-01, 6.40501495e-01, 7.91640206e-01],
           [3.85074989e-01, 6.44544624e-01, 7.92195800e-01],
           [3.91899948e-01, 6.48579321e-01, 7.92785568e-01],
           [3.98739225e-01, 6.52606122e-01, 7.93409493e-01],
           [4.05591483e-01, 6.56625563e-01, 7.94068296e-01],
           [4.12455751e-01, 6.60638156e-01, 7.94762350e-01],
           [4.19331270e-01, 6.64644391e-01, 7.95491889e-01],
           [4.26217137e-01, 6.68644756e-01, 7.96257468e-01],
           [4.33112697e-01, 6.72639715e-01, 7.97059421e-01],
           [4.40017472e-01, 6.76629713e-01, 7.97897951e-01],
           [4.46930955e-01, 6.80615189e-01, 7.98773396e-01],
           [4.53852643e-01, 6.84596572e-01, 7.99686162e-01],
           [4.60782169e-01, 6.88574277e-01, 8.00636552e-01],
           [4.67719305e-01, 6.92548699e-01, 8.01624758e-01],
           [4.74663794e-01, 6.96520228e-01, 8.02651067e-01],
           [4.81615423e-01, 7.00489244e-01, 8.03715761e-01],
           [4.88574025e-01, 7.04456118e-01, 8.04819112e-01],
           [4.95539466e-01, 7.08421212e-01, 8.05961390e-01],
           [5.02511648e-01, 7.12384878e-01, 8.07142859e-01],
           [5.09490468e-01, 7.16347464e-01, 8.08363820e-01],
           [5.16475923e-01, 7.20309302e-01, 8.09624485e-01],
           [5.23467994e-01, 7.24270721e-01, 8.10925111e-01],
           [5.30466680e-01, 7.28232045e-01, 8.12265957e-01],
           [5.37471997e-01, 7.32193588e-01, 8.13647284e-01],
           [5.44483974e-01, 7.36155662e-01, 8.15069356e-01],
           [5.51502655e-01, 7.40118572e-01, 8.16532436e-01],
           [5.58528093e-01, 7.44082616e-01, 8.18036793e-01],
           [5.65560351e-01, 7.48048092e-01, 8.19582695e-01],
           [5.72599494e-01, 7.52015291e-01, 8.21170421e-01],
           [5.79645572e-01, 7.55984504e-01, 8.22800274e-01],
           [5.86698691e-01, 7.59956011e-01, 8.24472509e-01],
           [5.93758934e-01, 7.63930095e-01, 8.26187410e-01],
           [6.00826384e-01, 7.67907032e-01, 8.27945267e-01],
           [6.07901129e-01, 7.71887100e-01, 8.29746373e-01],
           [6.14983250e-01, 7.75870574e-01, 8.31591029e-01],
           [6.22072834e-01, 7.79857725e-01, 8.33479537e-01],
           [6.29169958e-01, 7.83848826e-01, 8.35412205e-01],
           [6.36274700e-01, 7.87844149e-01, 8.37389350e-01],
           [6.43387067e-01, 7.91843978e-01, 8.39411351e-01],
           [6.50507192e-01, 7.95848572e-01, 8.41478467e-01],
           [6.57635139e-01, 7.99858201e-01, 8.43591026e-01],
           [6.64770965e-01, 8.03873137e-01, 8.45749356e-01],
           [6.71914719e-01, 8.07893655e-01, 8.47953794e-01],
           [6.79066400e-01, 8.11920038e-01, 8.50204719e-01],
           [6.86226039e-01, 8.15952565e-01, 8.52502471e-01],
           [6.93393708e-01, 8.19991503e-01, 8.54847355e-01],
           [7.00569422e-01, 8.24037135e-01, 8.57239719e-01],
           [7.07753182e-01, 8.28089747e-01, 8.59679918e-01],
           [7.14944891e-01, 8.32149648e-01, 8.62168381e-01],
           [7.22144634e-01, 8.36217106e-01, 8.64705376e-01],
           [7.29352388e-01, 8.40292417e-01, 8.67291253e-01],
           [7.36568084e-01, 8.44375889e-01, 8.69926392e-01],
           [7.43791613e-01, 8.48467840e-01, 8.72611193e-01],
           [7.51023008e-01, 8.52568559e-01, 8.75345929e-01],
           [7.58262202e-01, 8.56678357e-01, 8.78130945e-01],
           [7.65508972e-01, 8.60797591e-01, 8.80966691e-01],
           [7.72763339e-01, 8.64926558e-01, 8.83853414e-01],
           [7.80025213e-01, 8.69065585e-01, 8.86791435e-01],
           [7.87294319e-01, 8.73215052e-01, 8.89781195e-01],
           [7.94570638e-01, 8.77375274e-01, 8.92822926e-01],
           [8.01854042e-01, 8.81546599e-01, 8.95916926e-01],
           [8.09144173e-01, 8.85729441e-01, 8.99063637e-01],
           [8.16441048e-01, 8.89924117e-01, 9.02263216e-01],
           [8.23744365e-01, 8.94131035e-01, 9.05516019e-01],
           [8.31053886e-01, 8.98350590e-01, 9.08822338e-01],
           [8.38369506e-01, 9.02583145e-01, 9.12182351e-01],
           [8.45690727e-01, 9.06829180e-01, 9.15596469e-01],
           [8.53017550e-01, 9.11089037e-01, 9.19064754e-01],
           [8.60349445e-01, 9.15363219e-01, 9.22587577e-01],
           [8.67686293e-01, 9.19652113e-01, 9.26165022e-01],
           [8.75027600e-01, 9.23956223e-01, 9.29797371e-01],
           [8.82373139e-01, 9.28275982e-01, 9.33484713e-01],
           [8.89722392e-01, 9.32611918e-01, 9.37227270e-01],
           [8.97075091e-01, 9.36964489e-01, 9.41025082e-01],
           [9.04430596e-01, 9.41334277e-01, 9.44878356e-01],
           [9.11788680e-01, 9.45721742e-01, 9.48787026e-01],
           [9.19148458e-01, 9.50127565e-01, 9.52751328e-01],
           [9.26509697e-01, 9.54552226e-01, 9.56771099e-01],
           [9.33871535e-01, 9.58996420e-01, 9.60846440e-01],
           [9.41233327e-01, 9.63460782e-01, 9.64977268e-01],
           [9.48594410e-01, 9.67945970e-01, 9.69163437e-01],
           [9.55953566e-01, 9.72452838e-01, 9.73404963e-01],
           [9.63310039e-01, 9.76982106e-01, 9.77701539e-01],
           [9.70662646e-01, 9.81534657e-01, 9.82052927e-01],
           [9.78009723e-01, 9.86111562e-01, 9.86458936e-01],
           [9.85349764e-01, 9.90713869e-01, 9.90919110e-01],
           [9.92680732e-01, 9.95342843e-01, 9.95432977e-01],
           [1.00000000e+00, 1.00000000e+00, 1.00000000e+00]]
test_cm = ListedColormap(cm_data, name="arctic")

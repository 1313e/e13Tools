
from matplotlib.colors import ListedColormap

cm_type = "linear"

cm_data = [[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
           [1.86928655e-04, 2.43138159e-04, 3.27218414e-04],
           [6.17924867e-04, 8.49591308e-04, 1.19960729e-03],
           [1.22235499e-03, 1.77208520e-03, 2.61491023e-03],
           [1.96125525e-03, 2.99120651e-03, 4.59723106e-03],
           [2.80517140e-03, 4.49550357e-03, 7.17986247e-03],
           [3.73131551e-03, 6.27697643e-03, 1.03973079e-02],
           [4.71979735e-03, 8.32976070e-03, 1.42880218e-02],
           [5.75150296e-03, 1.06494644e-02, 1.88969888e-02],
           [6.81108698e-03, 1.32321645e-02, 2.42665934e-02],
           [7.88387143e-03, 1.60745477e-02, 3.04432478e-02],
           [8.95368843e-03, 1.91739208e-02, 3.74832093e-02],
           [1.00100332e-02, 2.25271425e-02, 4.51831287e-02],
           [1.10375376e-02, 2.61317910e-02, 5.29551051e-02],
           [1.20267872e-02, 2.99847980e-02, 6.07871426e-02],
           [1.29641064e-02, 3.40834734e-02, 6.86938142e-02],
           [1.38412915e-02, 3.84244667e-02, 7.66772344e-02],
           [1.46447133e-02, 4.29169391e-02, 8.47522474e-02],
           [1.53691294e-02, 4.73442978e-02, 9.29167793e-02],
           [1.60005691e-02, 5.17133230e-02, 1.01187346e-01],
           [1.65342073e-02, 5.60256464e-02, 1.09563482e-01],
           [1.69632117e-02, 6.02825656e-02, 1.18050263e-01],
           [1.72763379e-02, 6.44849302e-02, 1.26661594e-01],
           [1.74706913e-02, 6.86330039e-02, 1.35398184e-01],
           [1.75420027e-02, 7.27266960e-02, 1.44265130e-01],
           [1.74862066e-02, 7.67654855e-02, 1.53269111e-01],
           [1.73005743e-02, 8.07484127e-02, 1.62416765e-01],
           [1.69839661e-02, 8.46740750e-02, 1.71714652e-01],
           [1.65371312e-02, 8.85406191e-02, 1.81169201e-01],
           [1.59630635e-02, 9.23457280e-02, 1.90786639e-01],
           [1.52674187e-02, 9.60866051e-02, 2.00572891e-01],
           [1.44590033e-02, 9.97599564e-02, 2.10533455e-01],
           [1.35503414e-02, 1.03361972e-01, 2.20673244e-01],
           [1.25583286e-02, 1.06888311e-01, 2.30996381e-01],
           [1.14959728e-02, 1.10332805e-01, 2.41520182e-01],
           [1.03974825e-02, 1.13690483e-01, 2.52237429e-01],
           [9.28915482e-03, 1.16953470e-01, 2.63163477e-01],
           [8.21698618e-03, 1.20114774e-01, 2.74294116e-01],
           [7.22519570e-03, 1.23164083e-01, 2.85643610e-01],
           [6.37773642e-03, 1.26091216e-01, 2.97211766e-01],
           [5.75192008e-03, 1.28884405e-01, 3.08998057e-01],
           [5.44127379e-03, 1.31529734e-01, 3.21002423e-01],
           [5.55945613e-03, 1.34010016e-01, 3.33227562e-01],
           [6.25591456e-03, 1.36310142e-01, 3.45648205e-01],
           [7.70467683e-03, 1.38406647e-01, 3.58262101e-01],
           [1.01312711e-02, 1.40280153e-01, 3.71025295e-01],
           [1.38102258e-02, 1.41907074e-01, 3.83889838e-01],
           [1.90780500e-02, 1.43267722e-01, 3.96769070e-01],
           [2.63341021e-02, 1.44348688e-01, 4.09534541e-01],
           [3.60237808e-02, 1.45153978e-01, 4.21992749e-01],
           [4.80165668e-02, 1.45721575e-01, 4.33868506e-01],
           [6.07898836e-02, 1.46133738e-01, 4.44827987e-01],
           [7.39475520e-02, 1.46524112e-01, 4.54526221e-01],
           [8.71218213e-02, 1.47046972e-01, 4.62717863e-01],
           [9.99890057e-02, 1.47831540e-01, 4.69333758e-01],
           [1.12342422e-01, 1.48948092e-01, 4.74476166e-01],
           [1.24097180e-01, 1.50408211e-01, 4.78347113e-01],
           [1.35252731e-01, 1.52185008e-01, 4.81175102e-01],
           [1.45852093e-01, 1.54234045e-01, 4.83172070e-01],
           [1.55956010e-01, 1.56506614e-01, 4.84516825e-01],
           [1.65627029e-01, 1.58957193e-01, 4.85353487e-01],
           [1.74921086e-01, 1.61546693e-01, 4.85796866e-01],
           [1.83887781e-01, 1.64242290e-01, 4.85937392e-01],
           [1.92571397e-01, 1.67016705e-01, 4.85845203e-01],
           [2.01008125e-01, 1.69848281e-01, 4.85576369e-01],
           [2.09227631e-01, 1.72719910e-01, 4.85176284e-01],
           [2.17257093e-01, 1.75617479e-01, 4.84679735e-01],
           [2.25118903e-01, 1.78529947e-01, 4.84115110e-01],
           [2.32831971e-01, 1.81448618e-01, 4.83505541e-01],
           [2.40413773e-01, 1.84366326e-01, 4.82868600e-01],
           [2.47879222e-01, 1.87277430e-01, 4.82218613e-01],
           [2.55239935e-01, 1.90177798e-01, 4.81568659e-01],
           [2.62508190e-01, 1.93063668e-01, 4.80927241e-01],
           [2.69693791e-01, 1.95932291e-01, 4.80302265e-01],
           [2.76806145e-01, 1.98781308e-01, 4.79699343e-01],
           [2.83852859e-01, 2.01609039e-01, 4.79123858e-01],
           [2.90842425e-01, 2.04413708e-01, 4.78578169e-01],
           [2.97781027e-01, 2.07194251e-01, 4.78065699e-01],
           [3.04675066e-01, 2.09949607e-01, 4.77588183e-01],
           [3.11530905e-01, 2.12678735e-01, 4.77146122e-01],
           [3.18353984e-01, 2.15380861e-01, 4.76740066e-01],
           [3.25149431e-01, 2.18055283e-01, 4.76369989e-01],
           [3.31921818e-01, 2.20701445e-01, 4.76035717e-01],
           [3.38676047e-01, 2.23318643e-01, 4.75735885e-01],
           [3.45416539e-01, 2.25906276e-01, 4.75469042e-01],
           [3.52147432e-01, 2.28463785e-01, 4.75233469e-01],
           [3.58872667e-01, 2.30990622e-01, 4.75027114e-01],
           [3.65596000e-01, 2.33486249e-01, 4.74847618e-01],
           [3.72321010e-01, 2.35950136e-01, 4.74692343e-01],
           [3.79051105e-01, 2.38381761e-01, 4.74558390e-01],
           [3.85789527e-01, 2.40780612e-01, 4.74442621e-01],
           [3.92539350e-01, 2.43146190e-01, 4.74341671e-01],
           [3.99303486e-01, 2.45478011e-01, 4.74251972e-01],
           [4.06084680e-01, 2.47775613e-01, 4.74169763e-01],
           [4.12885513e-01, 2.50038557e-01, 4.74091112e-01],
           [4.19708394e-01, 2.52266435e-01, 4.74011929e-01],
           [4.26555566e-01, 2.54458870e-01, 4.73927985e-01],
           [4.33429096e-01, 2.56615529e-01, 4.73834929e-01],
           [4.40330877e-01, 2.58736118e-01, 4.73728308e-01],
           [4.47262627e-01, 2.60820395e-01, 4.73603583e-01],
           [4.54225877e-01, 2.62868171e-01, 4.73456159e-01],
           [4.61222001e-01, 2.64879304e-01, 4.73281370e-01],
           [4.68252189e-01, 2.66853714e-01, 4.73074531e-01],
           [4.75317443e-01, 2.68791384e-01, 4.72830966e-01],
           [4.82418590e-01, 2.70692365e-01, 4.72546012e-01],
           [4.89556284e-01, 2.72556770e-01, 4.72215041e-01],
           [4.96731007e-01, 2.74384780e-01, 4.71833479e-01],
           [5.03943078e-01, 2.76176641e-01, 4.71396825e-01],
           [5.11192656e-01, 2.77932667e-01, 4.70900666e-01],
           [5.18479745e-01, 2.79653235e-01, 4.70340691e-01],
           [5.25804207e-01, 2.81338789e-01, 4.69712711e-01],
           [5.33165760e-01, 2.82989834e-01, 4.69012666e-01],
           [5.40563996e-01, 2.84606937e-01, 4.68236639e-01],
           [5.47998381e-01, 2.86190723e-01, 4.67380863e-01],
           [5.55468268e-01, 2.87741878e-01, 4.66441730e-01],
           [5.62972906e-01, 2.89261140e-01, 4.65415798e-01],
           [5.70511444e-01, 2.90749302e-01, 4.64299792e-01],
           [5.78082942e-01, 2.92207212e-01, 4.63090606e-01],
           [5.85686377e-01, 2.93635766e-01, 4.61785310e-01],
           [5.93320652e-01, 2.95035913e-01, 4.60381140e-01],
           [6.00984600e-01, 2.96408651e-01, 4.58875503e-01],
           [6.08676990e-01, 2.97755031e-01, 4.57265973e-01],
           [6.16396534e-01, 2.99076153e-01, 4.55550284e-01],
           [6.24141888e-01, 3.00373170e-01, 4.53726328e-01],
           [6.31911656e-01, 3.01647292e-01, 4.51792150e-01],
           [6.39704395e-01, 3.02899787e-01, 4.49745941e-01],
           [6.47518790e-01, 3.04131862e-01, 4.47585727e-01],
           [6.55353420e-01, 3.05344827e-01, 4.45309748e-01],
           [6.63206483e-01, 3.06540294e-01, 4.42916952e-01],
           [6.71076345e-01, 3.07719805e-01, 4.40406039e-01],
           [6.78961321e-01, 3.08884987e-01, 4.37775825e-01],
           [6.86860444e-01, 3.10037005e-01, 4.35023779e-01],
           [6.94771257e-01, 3.11178142e-01, 4.32150111e-01],
           [7.02691883e-01, 3.12310341e-01, 4.29153974e-01],
           [7.10621369e-01, 3.13434936e-01, 4.26032613e-01],
           [7.18556935e-01, 3.14554671e-01, 4.22786803e-01],
           [7.26497008e-01, 3.15671491e-01, 4.19414832e-01],
           [7.34439562e-01, 3.16787760e-01, 4.15915743e-01],
           [7.42382295e-01, 3.17906153e-01, 4.12289032e-01],
           [7.50323101e-01, 3.19029298e-01, 4.08533575e-01],
           [7.58259755e-01, 3.20160025e-01, 4.04648259e-01],
           [7.66189246e-01, 3.21301896e-01, 4.00633524e-01],
           [7.74109849e-01, 3.22457608e-01, 3.96486487e-01],
           [7.82018215e-01, 3.23631253e-01, 3.92207708e-01],
           [7.89911340e-01, 3.24826817e-01, 3.87796578e-01],
           [7.97786176e-01, 3.26048478e-01, 3.83252108e-01],
           [8.05639429e-01, 3.27300784e-01, 3.78573367e-01],
           [8.13467529e-01, 3.28588684e-01, 3.73759504e-01],
           [8.21266587e-01, 3.29917578e-01, 3.68809794e-01],
           [8.29032361e-01, 3.31293362e-01, 3.63723677e-01],
           [8.36760212e-01, 3.32722474e-01, 3.58500815e-01],
           [8.44445054e-01, 3.34211948e-01, 3.53141159e-01],
           [8.52081303e-01, 3.35769466e-01, 3.47645025e-01],
           [8.59663680e-01, 3.37402735e-01, 3.42010184e-01],
           [8.67184950e-01, 3.39121300e-01, 3.36239378e-01],
           [8.74638457e-01, 3.40934537e-01, 3.30331451e-01],
           [8.82015792e-01, 3.42853509e-01, 3.24289566e-01],
           [8.89308693e-01, 3.44889474e-01, 3.18113772e-01],
           [8.96507430e-01, 3.47055153e-01, 3.11807046e-01],
           [9.03601159e-01, 3.49364427e-01, 3.05373917e-01],
           [9.10578045e-01, 3.51832233e-01, 2.98819687e-01],
           [9.17425306e-01, 3.54474524e-01, 2.92149441e-01],
           [9.24128013e-01, 3.57309034e-01, 2.85374670e-01],
           [9.30670206e-01, 3.60354430e-01, 2.78507034e-01],
           [9.37034124e-01, 3.63630705e-01, 2.71562970e-01],
           [9.43200523e-01, 3.67158878e-01, 2.64561284e-01],
           [9.49148056e-01, 3.70960979e-01, 2.57530187e-01],
           [9.54854012e-01, 3.75059432e-01, 2.50502284e-01],
           [9.60294188e-01, 3.79476561e-01, 2.43519942e-01],
           [9.65443495e-01, 3.84233624e-01, 2.36636376e-01],
           [9.70276633e-01, 3.89349969e-01, 2.29914017e-01],
           [9.74769072e-01, 3.94841430e-01, 2.23428910e-01],
           [9.78898458e-01, 4.00718736e-01, 2.17269710e-01],
           [9.82646014e-01, 4.06986169e-01, 2.11535247e-01],
           [9.85998282e-01, 4.13639913e-01, 2.06333005e-01],
           [9.88948593e-01, 4.20667166e-01, 2.01773983e-01],
           [9.91497714e-01, 4.28046475e-01, 1.97965284e-01],
           [9.93654475e-01, 4.35748051e-01, 1.95004287e-01],
           [9.95434916e-01, 4.43735889e-01, 1.92970673e-01],
           [9.96860996e-01, 4.51970062e-01, 1.91921144e-01],
           [9.97959093e-01, 4.60408953e-01, 1.91886616e-01],
           [9.98757744e-01, 4.69011962e-01, 1.92871494e-01],
           [9.99286105e-01, 4.77741074e-01, 1.94856249e-01],
           [9.99572766e-01, 4.86561852e-01, 1.97801551e-01],
           [9.99644642e-01, 4.95444189e-01, 2.01653231e-01],
           [9.99526328e-01, 5.04362497e-01, 2.06347533e-01],
           [9.99239798e-01, 5.13295546e-01, 2.11815930e-01],
           [9.98805120e-01, 5.22225413e-01, 2.17988686e-01],
           [9.98239367e-01, 5.31138095e-01, 2.24798160e-01],
           [9.97558375e-01, 5.40021678e-01, 2.32180021e-01],
           [9.96775076e-01, 5.48867473e-01, 2.40075470e-01],
           [9.95901448e-01, 5.57668183e-01, 2.48430786e-01],
           [9.94947957e-01, 5.66418205e-01, 2.57197883e-01],
           [9.93923772e-01, 5.75113312e-01, 2.66334162e-01],
           [9.92837278e-01, 5.83750154e-01, 2.75801869e-01],
           [9.91696427e-01, 5.92325947e-01, 2.85567413e-01],
           [9.90506865e-01, 6.00839694e-01, 2.95602866e-01],
           [9.89276374e-01, 6.09289056e-01, 3.05881199e-01],
           [9.88009409e-01, 6.17674057e-01, 3.16381417e-01],
           [9.86711623e-01, 6.25993968e-01, 3.27083635e-01],
           [9.85388780e-01, 6.34248074e-01, 3.37969803e-01],
           [9.84045177e-01, 6.42436628e-01, 3.49025197e-01],
           [9.82685421e-01, 6.50559712e-01, 3.60236258e-01],
           [9.81313992e-01, 6.58617506e-01, 3.71590844e-01],
           [9.79935195e-01, 6.66610309e-01, 3.83078111e-01],
           [9.78553263e-01, 6.74538480e-01, 3.94688243e-01],
           [9.77172807e-01, 6.82402187e-01, 4.06411767e-01],
           [9.75797342e-01, 6.90202206e-01, 4.18241233e-01],
           [9.74430772e-01, 6.97939097e-01, 4.30169336e-01],
           [9.73077521e-01, 7.05613164e-01, 4.42188636e-01],
           [9.71741539e-01, 7.13224967e-01, 4.54292716e-01],
           [9.70426224e-01, 7.20775339e-01, 4.66476262e-01],
           [9.69136506e-01, 7.28264377e-01, 4.78732312e-01],
           [9.67875246e-01, 7.35693180e-01, 4.91056873e-01],
           [9.66647295e-01, 7.43061911e-01, 5.03443599e-01],
           [9.65456004e-01, 7.50371439e-01, 5.15888320e-01],
           [9.64305243e-01, 7.57622395e-01, 5.28386373e-01],
           [9.63199712e-01, 7.64815063e-01, 5.40932108e-01],
           [9.62142592e-01, 7.71950391e-01, 5.53522065e-01],
           [9.61137904e-01, 7.79028968e-01, 5.66151754e-01],
           [9.60189756e-01, 7.86051354e-01, 5.78816637e-01],
           [9.59302347e-01, 7.93018092e-01, 5.91512107e-01],
           [9.58479227e-01, 7.99929984e-01, 6.04234519e-01],
           [9.57724347e-01, 8.06787674e-01, 6.16979704e-01],
           [9.57041640e-01, 8.13591824e-01, 6.29743535e-01],
           [9.56435015e-01, 8.20343104e-01, 6.42521925e-01],
           [9.55908364e-01, 8.27042197e-01, 6.55310810e-01],
           [9.55465560e-01, 8.33689798e-01, 6.68106140e-01],
           [9.55110456e-01, 8.40286605e-01, 6.80903874e-01],
           [9.54846893e-01, 8.46833326e-01, 6.93699966e-01],
           [9.54678703e-01, 8.53330667e-01, 7.06490357e-01],
           [9.54609713e-01, 8.59779335e-01, 7.19270963e-01],
           [9.54643764e-01, 8.66180030e-01, 7.32037667e-01],
           [9.54784720e-01, 8.72533436e-01, 7.44786302e-01],
           [9.55036760e-01, 8.78840141e-01, 7.57512192e-01],
           [9.55403740e-01, 8.85100814e-01, 7.70211225e-01],
           [9.55889661e-01, 8.91316067e-01, 7.82879155e-01],
           [9.56498802e-01, 8.97486407e-01, 7.95511435e-01],
           [9.57235859e-01, 9.03612203e-01, 8.08103054e-01],
           [9.58105959e-01, 9.09693670e-01, 8.20648650e-01],
           [9.59114178e-01, 9.15730973e-01, 8.33143625e-01],
           [9.60266799e-01, 9.21723882e-01, 8.45582205e-01],
           [9.61571664e-01, 9.27671665e-01, 8.57957236e-01],
           [9.63037288e-01, 9.33573229e-01, 8.70262672e-01],
           [9.64675635e-01, 9.39426385e-01, 8.82489494e-01],
           [9.66502466e-01, 9.45227633e-01, 8.94626877e-01],
           [9.68539680e-01, 9.50971421e-01, 9.06660521e-01],
           [9.70820283e-01, 9.56648824e-01, 9.18566393e-01],
           [9.73392449e-01, 9.62246490e-01, 9.30306084e-01],
           [9.76327429e-01, 9.67745910e-01, 9.41803948e-01],
           [9.79709978e-01, 9.73129849e-01, 9.52922661e-01],
           [9.83580251e-01, 9.78407163e-01, 9.63445684e-01],
           [9.87802079e-01, 9.83651860e-01, 9.73201103e-01],
           [9.92061826e-01, 9.88977271e-01, 9.82308956e-01],
           [9.96134486e-01, 9.94434448e-01, 9.91141471e-01],
           [1.00000000e+00, 1.00000000e+00, 1.00000000e+00]]
test_cm = ListedColormap(cm_data, name="heat")

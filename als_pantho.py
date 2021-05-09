import numpy as np
import pandas as pd
from numpy.random import shuffle
from numpy.linalg import inv


def preProcess():
    col = []
    for i in range(101):
        col.append(i)
    df = pd.read_csv('./datasets/data.csv', names=col, header=None)
    train_df = pd.read_csv('./datasets/data.csv', names=col, header=None)
    test_df = pd.read_csv('./datasets/data.csv', names=col, header=None)
    val_df = pd.read_csv('./datasets/data.csv', names=col, header=None)
    va = df.shape[0]
    #va = 1000
    df = df[:va]
    train_df = train_df[:va]
    test_df = test_df[:va]
    val_df = val_df[:va]
    for i in range(int(df.shape[0])):
        if i % 100 == 0:
            print(i)
        vals = []
        for j in col:
            if -10 <= df.iloc[i][j] <= 10 and j != 0:
                vals.append(j)
        vals = np.array(vals)
        shuffle(vals)
        vals = vals.tolist()
        test = vals[:int(0.2 * len(vals))]
        val = vals[-int(0.2 * len(vals)):]
        for j in vals:
            if j in test:
                train_df.iat[i, j] = 90.0
                val_df.iat[i, j] = 90.0
            elif j in val:
                train_df.iat[i, j] = 90.0
                test_df.iat[i, j] = 90.0
            else:
                test_df.iat[i, j] = 90.0
                val_df.iat[i, j] = 90.0
    train_df.drop([0], axis=1, inplace=True)
    test_df.drop([0], axis=1, inplace=True)
    val_df.drop([0], axis=1, inplace=True)
    return train_df, test_df, val_df


train_df, test_df, val_df = preProcess()
ks = [2, 3, 5]
ls = [10.0, 50.0, 100.0, 1000.0]
K = 5
lemda_u = 10.0
lemda_v = 10.0
U = np.random.rand(train_df.shape[0], K)
V = np.random.rand(K, train_df.shape[1])
print(train_df.shape)


def error(df, U, V):
    sm = 0
    cnt = 0
    for i in range(len(df)):
        for j in range(df.shape[1]):
            if -10 <= df.iloc[i][j + 1] <= 10:
                # V = np.transpose(V)
                pred = np.dot(U[i], V[:, j])
                sm += (df.iloc[i][j + 1] - pred) ** 2
                cnt += 1
                # V = np.transpose(V)
    return (sm / cnt) ** 0.5


# error(test_df)

def ALS(df):
    global U
    global V
    it = 0
    prev = 20
    now = 20
    while (True):
        # update V
        # print(U.shape, V.shape)
        for i in range(len(V[0])):
            # print(V.shape)
            sum_u = np.zeros(shape=(K, K))
            sum_x = np.zeros(K)
            for j in range(len(U)):
                if -10 <= df.iloc[j][i + 1] <= 10:
                    sum_u += np.dot(np.transpose([U[j]]), [U[j]])
                    sum_x += U[j] * df.iloc[j][i + 1]
            # V = np.transpose(V)
            V[:, i] = np.dot(inv(sum_u + lemda_u * np.identity(K)), sum_x)
            # V = np.transpose(V)
            # print(U.shape, V.shape)
        # print(it)
        # update U
        # V = np.transpose(V)
        for i in range(len(U)):
            sum_u = np.zeros(shape=(K, K))
            sum_x = np.zeros(K)
            for j in range(df.shape[1]):
                # print(i, j)
                if -10 <= df.iloc[i][j + 1] <= 10:
                    sum_u += np.dot(np.transpose([V[:, j]]), [V[:, j]])
                    sum_x += V[:, j] * df.iloc[i][j + 1]
            U[i] = np.dot(inv(sum_u + lemda_v * np.identity(K)), sum_x)
        # V = np.transpose(V)
        it += 1
        now = error(train_df, U, V)
        # print(now)
        if abs(prev - now) / prev <= 0.05:
            print("k = ", K, "lemda_u = ", lemda_u, "lemda_v = ", lemda_v, "train error =", now, "val error = ",
                  error(val_df, U, V))
            break
        prev = now

'''
#for hyper parameter tuning

print("in tuning")
for kk in ks:
    for lus in ls:
        for lvs in ls:
            global U, V, K, lemda_v, lemda_u
            K = kk
            lemda_u = lus
            lemda_v = lvs
            U = np.random.rand(train_df.shape[0], K)
            V = np.random.rand(K, train_df.shape[1])
            ALS(train_df)
'''
#for deployment

def ALS_Deploy(df, U, V):
    it = 0
    prev = 20
    while (True):
        # update V
        for i in range(len(V[0])):
            sum_u = np.zeros(shape=(K, K))
            sum_x = np.zeros(K)
            for j in range(len(U)):
                if -10 <= df.iloc[j][i + 1] <= 10:
                    sum_u += np.dot(np.transpose([U[j]]), [U[j]])
                    sum_x += U[j] * df.iloc[j][i + 1]
            V[:, i] = np.dot(inv(sum_u + lemda_u * np.identity(K)), sum_x)
        # update U
        for i in range(len(U)):
            sum_u = np.zeros(shape=(K, K))
            sum_x = np.zeros(K)
            for j in range(df.shape[1]):
                if -10 <= df.iloc[i][j + 1] <= 10:
                    sum_u += np.dot(np.transpose([V[:, j]]), [V[:, j]])
                    sum_x += V[:, j] * df.iloc[i][j + 1]
            U[i] = np.dot(inv(sum_u + lemda_v * np.identity(K)), sum_x)
        it += 1
        now = error(df, U, V)
        if abs(prev - now) / prev <= 0.05:
            print("k = ", K, "lemda_u = ", lemda_u, "lemda_v = ", lemda_v, "train error =", now)
            return U, V
        prev = now

print("in deployment")
K = 3
lemda_u = 100.0
lemda_v = 10.0
U = np.random.rand(train_df.shape[0], K)
V = np.random.rand(K, train_df.shape[1])
U, V = ALS_Deploy(train_df, U, V)
np.save("best_U.npy", U)
np.save("best_V.npy", V)
U = np.load("best_U.npy")
V = np.load("best_V.npy")
U, V = ALS_Deploy(val_df, U, V)
np.save("bestD_U.npy", U)
np.save("bestD_V.npy", V)
print("error on test set = ", error(test_df, U, V))













'''
training 3.4934246534579807
k =  10 lemda_u =  0.01 lemda_v =  0.01 val error =  4.6232609342394735

k =  5 lemda_u =  0.01 lemda_v =  0.01 train error = 3.784733837786191 val error =  4.453338351641757
k =  5 lemda_u =  0.01 lemda_v =  0.1 train error = 3.787407682167172 val error =  4.459072977788265
k =  5 lemda_u =  0.01 lemda_v =  1.0 train error = 3.805323067560186 val error =  4.4159418116902085
k =  5 lemda_u =  0.01 lemda_v =  10.0 train error = 3.8066867942792384 val error =  4.370597391832547
k =  5 lemda_u =  0.1 lemda_v =  0.01 train error = 3.7706956620687926 val error =  4.396060209976194
k =  5 lemda_u =  0.1 lemda_v =  0.1 train error = 3.8022752931543953 val error =  4.510470193522076
k =  5 lemda_u =  0.1 lemda_v =  1.0 train error = 3.802427701338936 val error =  4.432100655754551
k =  5 lemda_u =  0.1 lemda_v =  10.0 train error = 3.799295188214345 val error =  4.3886348023904596
k =  5 lemda_u =  1.0 lemda_v =  0.01 train error = 3.7961222246466773 val error =  4.454709069994729
k =  5 lemda_u =  1.0 lemda_v =  0.1 train error = 3.7871853010144814 val error =  4.471100408774592
k =  5 lemda_u =  1.0 lemda_v =  1.0 train error = 3.7929302440664427 val error =  4.433501455889145
k =  5 lemda_u =  1.0 lemda_v =  10.0 train error = 3.786612236735318 val error =  4.346694016043941
k =  5 lemda_u =  10.0 lemda_v =  0.01 train error = 3.757777205384591 val error =  4.407642994225168
k =  5 lemda_u =  10.0 lemda_v =  0.1 train error = 3.7883694530031877 val error =  4.439155282520885
k =  5 lemda_u =  10.0 lemda_v =  1.0 train error = 3.7949973334063314 val error =  4.439268521281162
k =  5 lemda_u =  10.0 lemda_v =  10.0 train error = 3.7930542209459985 val error =  4.328726000432735          min
k =  10 lemda_u =  0.01 lemda_v =  0.01 train error = 3.409438775375747 val error =  4.957124518601975
k =  10 lemda_u =  0.01 lemda_v =  0.1 train error = 3.384928338217598 val error =  4.873470312288955
k =  10 lemda_u =  0.01 lemda_v =  1.0 train error = 3.358462229273164 val error =  4.820160893986011
k =  10 lemda_u =  0.01 lemda_v =  10.0 train error = 3.398356536199673 val error =  4.569938994007807
k =  10 lemda_u =  0.1 lemda_v =  0.01 train error = 3.3930234873776195 val error =  4.964679687383304
k =  10 lemda_u =  0.1 lemda_v =  0.1 train error = 3.3945552569871085 val error =  5.079855090036958
k =  10 lemda_u =  0.1 lemda_v =  1.0 train error = 3.3855683658326976 val error =  4.798540660136653
k =  10 lemda_u =  0.1 lemda_v =  10.0 train error = 3.3889008333914425 val error =  4.550275328232535
k =  10 lemda_u =  1.0 lemda_v =  0.01 train error = 3.3754919087205457 val error =  4.969423021231245
k =  10 lemda_u =  1.0 lemda_v =  0.1 train error = 3.3576977955152962 val error =  4.923378219684782
k =  10 lemda_u =  1.0 lemda_v =  1.0 train error = 3.3779782784918364 val error =  4.698091968527666
k =  10 lemda_u =  1.0 lemda_v =  10.0 train error = 3.391241869874733 val error =  4.600250672494715
k =  10 lemda_u =  10.0 lemda_v =  0.01 train error = 3.3610820672186317 val error =  4.847795300064393
k =  10 lemda_u =  10.0 lemda_v =  0.1 train error = 3.364421440525475 val error =  4.814719284301557
k =  10 lemda_u =  10.0 lemda_v =  1.0 train error = 3.412823225856318 val error =  4.75682734189427
k =  10 lemda_u =  10.0 lemda_v =  10.0 train error = 3.388242516050989 val error =  4.492046411981934
k =  20 lemda_u =  0.01 lemda_v =  0.01 train error = 2.6710682596721984 val error =  7.099763236565567
k =  20 lemda_u =  0.01 lemda_v =  0.1 train error = 2.668085191159473 val error =  6.407683171689399
k =  20 lemda_u =  0.01 lemda_v =  1.0 train error = 2.5793788914066225 val error =  5.707800048723905
k =  20 lemda_u =  0.01 lemda_v =  10.0 train error = 2.593198703394245 val error =  5.1671875073473865
k =  20 lemda_u =  0.1 lemda_v =  0.01 train error = 2.675265938086048 val error =  6.923506465934869
k =  20 lemda_u =  0.1 lemda_v =  0.1 train error = 2.655793291981736 val error =  6.327444341504308
k =  20 lemda_u =  0.1 lemda_v =  1.0 train error = 2.587162092048152 val error =  5.7167840299923105
k =  20 lemda_u =  0.1 lemda_v =  10.0 train error = 2.601157849297743 val error =  5.230778714669142
k =  20 lemda_u =  1.0 lemda_v =  0.01 train error = 2.6967607241169684 val error =  6.953463071861048
k =  20 lemda_u =  1.0 lemda_v =  0.1 train error = 2.6449598250853605 val error =  6.146798676223732
k =  20 lemda_u =  1.0 lemda_v =  1.0 train error = 2.5738592646737737 val error =  5.652335888220373
k =  20 lemda_u =  1.0 lemda_v =  10.0 train error = 2.590020903360535 val error =  5.189068687496589
k =  20 lemda_u =  10.0 lemda_v =  0.01 train error = 2.581864346303681 val error =  6.87673890593101
k =  20 lemda_u =  10.0 lemda_v =  0.1 train error = 2.5460968778401756 val error =  6.230915151655609
k =  20 lemda_u =  10.0 lemda_v =  1.0 train error = 2.581086638167025 val error =  5.4941108165840395
k =  20 lemda_u =  10.0 lemda_v =  10.0 train error = 2.6420887232182797 val error =  5.028771839463297
k =  40 lemda_u =  0.01 lemda_v =  0.01 train error = 1.235552228865607 val error =  11.912552369289147
k =  40 lemda_u =  0.01 lemda_v =  0.1 train error = 1.1591892871448304 val error =  9.45063707042613
k =  40 lemda_u =  0.01 lemda_v =  1.0 train error = 1.0972735488015646 val error =  7.615016704050372
k =  40 lemda_u =  0.01 lemda_v =  10.0 train error = 1.1109098953802221 val error =  6.482928127005981
k =  40 lemda_u =  0.1 lemda_v =  0.01 train error = 1.254130259581271 val error =  12.054744949872338
k =  40 lemda_u =  0.1 lemda_v =  0.1 train error = 1.154669442840175 val error =  9.511013624887465
k =  40 lemda_u =  0.1 lemda_v =  1.0 train error = 1.1404041996134313 val error =  7.438302535125972
k =  40 lemda_u =  0.1 lemda_v =  10.0 train error = 1.1354334798310013 val error =  6.410491854879672
k =  40 lemda_u =  1.0 lemda_v =  0.01 train error = 1.243853280890386 val error =  11.820973544999276
k =  40 lemda_u =  1.0 lemda_v =  0.1 train error = 1.1395388759746758 val error =  9.46566470224012
k =  40 lemda_u =  1.0 lemda_v =  1.0 train error = 1.131224630361325 val error =  7.2651797173313355
k =  40 lemda_u =  1.0 lemda_v =  10.0 train error = 1.1409816426996338 val error =  6.327671550061976
k =  40 lemda_u =  10.0 lemda_v =  0.01 train error = 1.132166664189478 val error =  10.535307573005865
k =  40 lemda_u =  10.0 lemda_v =  0.1 train error = 1.0984058827531178 val error =  8.32995653394441
k =  40 lemda_u =  10.0 lemda_v =  1.0 train error = 1.1477870578786518 val error =  6.7782004579795165
k =  40 lemda_u =  10.0 lemda_v =  10.0 train error = 1.3282247550652493 val error =  5.605626764144175

k =  2 lemda_u =  10.0 lemda_v =  10.0 train error = 4.183375385706686 val error =  4.382136970840334
k =  2 lemda_u =  10.0 lemda_v =  50.0 train error = 4.116247324470948 val error =  4.305615533807524
k =  2 lemda_u =  10.0 lemda_v =  100.0 train error = 4.122124215167907 val error =  4.306176172080842
k =  2 lemda_u =  10.0 lemda_v =  1000.0 train error = 4.233102608127446 val error =  4.364963938628149
k =  2 lemda_u =  50.0 lemda_v =  10.0 train error = 4.143679785153611 val error =  4.344319517761081
k =  2 lemda_u =  50.0 lemda_v =  50.0 train error = 4.13795212834822 val error =  4.307074872511654
k =  2 lemda_u =  50.0 lemda_v =  100.0 train error = 4.182436146684126 val error =  4.334753452284825
k =  2 lemda_u =  50.0 lemda_v =  1000.0 train error = 4.732421162424615 val error =  4.7706993228589845
k =  2 lemda_u =  100.0 lemda_v =  10.0 train error = 4.409962689538124 val error =  4.595846904734361
k =  2 lemda_u =  100.0 lemda_v =  50.0 train error = 4.1839124245089945 val error =  4.338127600162654
k =  2 lemda_u =  100.0 lemda_v =  100.0 train error = 4.565521917862568 val error =  4.631876431206914
k =  2 lemda_u =  100.0 lemda_v =  1000.0 train error = 5.074134520158357 val error =  5.077594318854761
k =  2 lemda_u =  1000.0 lemda_v =  10.0 train error = 4.232964682261541 val error =  4.36654378721911
k =  2 lemda_u =  1000.0 lemda_v =  50.0 train error = 4.731084447737124 val error =  4.769548835111855
k =  2 lemda_u =  1000.0 lemda_v =  100.0 train error = 5.07704950433972 val error =  5.080441671024438
k =  2 lemda_u =  1000.0 lemda_v =  1000.0 train error = 5.291744624895907 val error =  5.275371281712444
k =  3 lemda_u =  10.0 lemda_v =  10.0 train error = 4.003844767862967 val error =  4.318051968213631
k =  3 lemda_u =  10.0 lemda_v =  50.0 train error = 3.989648327677388 val error =  4.283899045268348
k =  3 lemda_u =  10.0 lemda_v =  100.0 train error = 4.033191206094964 val error =  4.306964603377332
k =  3 lemda_u =  10.0 lemda_v =  1000.0 train error = 4.217282991914765 val error =  4.3609596743629275
k =  3 lemda_u =  50.0 lemda_v =  10.0 train error = 4.016378580818472 val error =  4.336330795847452
k =  3 lemda_u =  50.0 lemda_v =  50.0 train error = 4.093943994828341 val error =  4.338665916260218
k =  3 lemda_u =  50.0 lemda_v =  100.0 train error = 4.065066104823718 val error =  4.282755773985273
k =  3 lemda_u =  50.0 lemda_v =  1000.0 train error = 4.72669615972861 val error =  4.765909677650307
k =  3 lemda_u =  100.0 lemda_v =  10.0 train error = 3.9871838770597914 val error =  4.281131579542728             min
k =  3 lemda_u =  100.0 lemda_v =  50.0 train error = 4.104074358993381 val error =  4.309341095274856
k =  3 lemda_u =  100.0 lemda_v =  100.0 train error = 4.211226835804446 val error =  4.3594662138864075
k =  3 lemda_u =  100.0 lemda_v =  1000.0 train error = 5.083560048823246 val error =  5.086062086294801
k =  3 lemda_u =  1000.0 lemda_v =  10.0 train error = 4.545736508614498 val error =  4.614907466803189
k =  3 lemda_u =  1000.0 lemda_v =  50.0 train error = 4.787659020373211 val error =  4.820092368137253
k =  3 lemda_u =  1000.0 lemda_v =  100.0 train error = 5.051183609076889 val error =  5.056852243248703
k =  3 lemda_u =  1000.0 lemda_v =  1000.0 train error = 5.291634158869817 val error =  5.275270868544176
k =  5 lemda_u =  10.0 lemda_v =  10.0 train error = 3.773790351303525 val error =  4.31271653607926
k =  5 lemda_u =  10.0 lemda_v =  50.0 train error = 3.835819035239833 val error =  4.363095159187275
k =  5 lemda_u =  10.0 lemda_v =  100.0 train error = 3.84423757435674 val error =  4.2920887425079055
k =  5 lemda_u =  10.0 lemda_v =  1000.0 train error = 4.183110084101491 val error =  4.348413849367782
k =  5 lemda_u =  50.0 lemda_v =  10.0 train error = 3.8208907241424415 val error =  4.337080302227798
k =  5 lemda_u =  50.0 lemda_v =  50.0 train error = 3.8985228299035186 val error =  4.289707167626135
k =  5 lemda_u =  50.0 lemda_v =  100.0 train error = 4.008143957840652 val error =  4.307713475744925
k =  5 lemda_u =  50.0 lemda_v =  1000.0 train error = 4.727961082035752 val error =  4.767089716033368
k =  5 lemda_u =  100.0 lemda_v =  10.0 train error = 3.8230741591688853 val error =  4.323198147600151
k =  5 lemda_u =  100.0 lemda_v =  50.0 train error = 4.002728504101874 val error =  4.295705621446019
k =  5 lemda_u =  100.0 lemda_v =  100.0 train error = 4.141390610943505 val error =  4.335762858542693
k =  5 lemda_u =  100.0 lemda_v =  1000.0 train error = 5.108749168127083 val error =  5.108963955572371
k =  5 lemda_u =  1000.0 lemda_v =  10.0 train error = 4.501900674830457 val error =  4.580371375365629
k =  5 lemda_u =  1000.0 lemda_v =  50.0 train error = 4.781466488511405 val error =  4.8144471164042475
k =  5 lemda_u =  1000.0 lemda_v =  100.0 train error = 5.037244981369786 val error =  5.044227028777786
k =  5 lemda_u =  1000.0 lemda_v =  1000.0 train error = 5.2915495338263385 val error =  5.275194011564836

full dataset
(24983, 100)
in deployment
k =  5 lemda_u =  10.0 lemda_v =  10.0 train error = 3.8205086878310897
k =  5 lemda_u =  10.0 lemda_v =  10.0 train error = 3.320921074838395
error on test set =  4.510477470282449

(24983, 100)
in deployment
k =  3 lemda_u =  100.0 lemda_v =  10.0 train error = 4.006442451992161
k =  3 lemda_u =  100.0 lemda_v =  10.0 train error = 3.6955737191637272
error on test set =  4.4529275505171455
'''

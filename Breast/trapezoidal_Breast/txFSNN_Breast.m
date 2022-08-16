%% Fuzzy spiking neural network, Iris, one by one, add testing, good, 0.02 %%
% M = 6; Q = 10; K = 5; L = 12; tau = 8; beta1 = 2; beta2 = 0.5; eta = 0.0085;

%save('txFSNN_Breast_jd.mat')
clear
close all
clc
P = 1;
Breast_cancer_wisconsin = load('Breast_FillAllValues.txt');

% datax = Iris_data_numerical(:,1:4);
datax1 = Breast_cancer_wisconsin(:,1:9);
datax = mapminmax(datax1',0,1);  % transform to [0,1]
datax = datax';
SampleNum = size(datax,1); % the number of samples
datay = Breast_cancer_wisconsin(:,10);
datay1 = zeros(SampleNum,P);
for j = 1:SampleNum
    if datay(j) == 0
        datay1(j,:) = 6;
    else
        datay1(j,:) = 8;
    end
end

pos = randperm(SampleNum);
datax = datax(pos,:);    %%% 训练集输入
datay = datay(pos,:);
datay1 = datay1(pos,:);

kkk=zeros(1,5);
nBreast_cancer_wisconsin=[datax datay datay1];
indices = crossvalind('Kfold', 699, 10);%将数据样本随机分割为10部分
for i = 1:10 %循环10次，分别取出第i部分作为测试样本，其余部分作为训练样本
    test = (indices == i);
    train = ~test;
    trainData = nBreast_cancer_wisconsin(train, :);
    x = trainData(:,1:9);
    J = size(x,1);      % the number of training samples
    O = trainData(:,10);
    O1 = trainData(:,11);
    
    testData = nBreast_cancer_wisconsin(test, :);
    testx = testData(:,1:9);
    testJ = size(testx,1); % the number of test samples
    testO= testData(:,10);
    testO1= testData(:,11);
    
    N = size(x,2);         % the number of features
    M = 6;                     % the number of Gaussian receptive fields
    C = N*M+1;                   % the number of neurons of Fuzzy Coding Layer
    Q = 10; % the number of hidden layer
    K = 5; %the number of sysnaptsis
    L = 12; % the interval
    tau = 8;
    beta1 = 2;
    beta2 = 0.5;
    eta = 0.0085;
    eta1 = eta;
    eta2= eta;
    % codingt = 100*ones(SampleNum,N*M); % the initial code t
    
    epoch = 1;
    max_epoch = 50;
    tF = zeros(1,C);
    % a = rand(1,C-1);
    % b = rand(1,C-1);
    a = zeros(1,C-1);
    b1 = zeros(1,C-1);
    b2 = zeros(1,C-1);
    for n=1:N   % Get the initial center and width of Gaussian
        MaxValue = max(x(:,n));
        MinValue = min(x(:,n));
        for m=1:M
            a(M*(n-1)+m) = MinValue+(2*m-3)*(MaxValue-MinValue)/(2*(M-2));
            b1(M*(n-1)+m) = (MaxValue-MinValue)/(beta1*(M-2));
            b2(M*(n-1)+m) = (MaxValue-MinValue)/(beta2*(M-2));
        end
    end
    w = 0.1.*rand(C*K,Q);
    u = 0.1.*rand(Q*K,P);
    D_u = zeros(Q*K,P);
    D_w = zeros(C*K,Q);
    theta = 1;
    err = zeros(1,J);
    Err = zeros(1,max_epoch-1);
    ACC = zeros(1,max_epoch-1);
    output = zeros(J,P);
    
    y1 = zeros(C*K,1);
    y = zeros(C*K,Q);
    D_y1 = zeros(C*K,1);
    D_y = zeros(C*K,Q);
    Y1 = zeros(Q*K,1);
    Y = zeros(Q*K,P);
    D_Y1 = zeros(Q*K,1);
    D_Y = zeros(Q*K,P);
    SH = zeros(1,Q);
    D_SH = zeros(1,Q);
    SO = zeros(1,P);
    D_SO = zeros(1,P);
    I=ones(1,Q);
    I(1)=-1;
    
    testy1 = zeros(C*K,1);
    testy = zeros(C*K,Q);
    D_testy1 = zeros(C*K,1);
    D_testy = zeros(C*K,Q);
    testY1 = zeros(Q*K,1);
    testY = zeros(Q*K,P);
    D_testY1 = zeros(Q*K,1);
    D_testY = zeros(Q*K,P);
    testSH = zeros(1,Q);
    D_testSH = zeros(1,Q);
    testSO = zeros(1,P);
    D_testSO = zeros(1,P);
    testerr = zeros(1,J);
    testErr = zeros(1,max_epoch-1);
    testACC = zeros(1,max_epoch-1);
    testACC1 = 0;
    ACC1 = 0;
    testoutput = zeros(J,P);
    T_interval = 4;
    deltaQ = zeros(1,Q);
    D_a = zeros(1,C-1);
    D_b = zeros(1,C-1);
    testtF = zeros(1,C);
    tmp_ACC = 0;
    tmp_testACC = 0;
    while epoch<max_epoch
        % while epoch<max_epoch && (ACC1<0.972 || testACC1<0.972)
        %%%%%%% training process %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        accuracy = 0;
        for j = 1:J
            tH = 100*ones(1,Q);
            tO = 100*ones(1,P);
            %%%%% coding process %%%%%%%%%%%%%%%%%%%%%
            for n = 1:N
                for m = 1:M
                    c = M*(n-1)+m;
                    tF(c) = roundn(T_interval.*(1-ftx(a(c),b1(c),b2(c),x(j,n))),-1);
                end
            end
            %%%%% Hidden Layer %%%%%%%%%%%%%%%%%%%%%
            for l = 0:0.2:L
                for c = 1:C
                    for k = 1:K
                        dk = k;
                        tmp = l-tF(c)-dk;
                        if tmp>0
                            y1((c-1)*K+k) = tmp/tau*exp(1-tmp/tau);
                            D_y1((c-1)*K+k) = (1-tmp/tau)/tau*exp(1-tmp/tau);
                        else
                            y1((c-1)*K+k) = 0;
                            D_y1((c-1)*K+k) = 0;
                        end
                    end
                end
                for q = 1:Q
                    if tH(q)==100
                        SH(q) = sum(w(:,q).*y1);
                        if SH(q)>theta
                            tH(q) = l;
                            y(:,q) = y1;
                            D_y(:,q) = D_y1;
                            D_SH(q) = sum(w(:,q).*D_y1);
                        end
                    end
                end
                if max(tH)<100
                    break
                end
            end
            
            %%%%% Output Layer %%%%%%%%%%%%%%
            for l = min(tH):0.2:L
                for q = 1:Q
                    for k = 1:K
                        dk = k;
                        tmp = l-tH(q)-dk;
                        if tmp>0
                            Y1((q-1)*K+k) = I(q)*tmp/tau*exp(1-tmp/tau);
                            D_Y1((q-1)*K+k) = I(q)*(1-tmp/tau)/tau*exp(1-tmp/tau);
                        else
                            Y1((q-1)*K+k) = 0;
                            D_Y1((q-1)*K+k) = 0;
                        end
                    end
                end
                for p = 1:P
                    if tO(p)==100
                        SO(p) = sum(u(:,p).*Y1);
                        if SO(p)>theta
                            tO(p) = l;
                            Y(:,p) = Y1;
                            D_Y(:,p) = D_Y1;
                            D_SO(p) = sum(u(:,p).*D_Y1);
                        end
                    end
                end
                if max(tO)<100
                    break
                end
            end
            
            %%%%%%%%%% err caculate %%%%%%%%%%%%%%%
            output(j,:) = tO;
            err(j) = sum((tO-O1(j,:)).^2)/(2*P);
            %%%%%%%%% training accuracy %%%%%%%%%%
            if tO<7
                pos = 0;
            else
                pos = 1;
            end
            if pos==O(j)
                accuracy=accuracy+1;
            end
            %%%%%%%%%% learning process %%%%%%%%%%%%%
            deltaP = (O1(j,:)-tO)./D_SO;
            for q = 1:Q
                deltaQ(q) = sum(deltaP.*(sum(u(K*(q-1)+1:K*q,:).*D_Y(K*(q-1)+1:K*q,:),1)))/D_SH(q);
            end
            for q = 1:Q
                for k = 1:K
                    D_u((q-1)*K+k,:) = deltaP.*Y((q-1)*K+k,:);
                end
            end
            for c = 1:C
                for k = 1:K
                    D_w((c-1)*K+k,:) = deltaQ.*y((c-1)*K+k,:);
                end
            end
            for n = 1:N
                for m = 1:M
                    c = (n-1)*M+m;
                    D_a(c) = +sum(deltaQ.*sum(w((c-1)*K+1:c*K,:).*D_y((c-1)*K+1:c*K,:),1))*(T_interval.*ftxa(a(c),b1(c),b2(c),x(j,n)));
                    D_b1(c) = +sum(deltaQ.*sum(w((c-1)*K+1:c*K,:).*D_y((c-1)*K+1:c*K,:),1))*(T_interval.*ftxb1(a(c),b1(c),b2(c),x(j,n)));
                    D_b2(c) = +sum(deltaQ.*sum(w((c-1)*K+1:c*K,:).*D_y((c-1)*K+1:c*K,:),1))*(T_interval.*ftxb2(a(c),b1(c),b2(c),x(j,n)));
                end
            end
            
            u = u-eta.*D_u;
            w = w-eta1.*D_w;
            a = a-eta2.*D_a;
            b1 = b1-eta2.*D_b1;
            b2 = b2-eta2.*D_b2;
            w(w<0)=0;
            u(u<0)=0;
        end
        %%%%%% testing process %%%%%%%%%%%%%
        testaccuracy = 0;
        for j = 1:testJ
            testtH = 100*ones(1,Q);
            testtO = 100*ones(1,P);
            %%%%% testing coding process %%%%%%%%%%%%%%%%%%%%%
            for n = 1:N
                for m = 1:M
                    c = M*(n-1)+m;
                    testtF(c) = roundn(T_interval.*(1-ftx(a(c),b1(c),b2(c),testx(j,n))),-1);
                end
            end
            %%%%% testing Hidden Layer %%%%%%%%%%%%%%%%%%%%%
            for l = 1:0.2:L
                for c = 1:C
                    for k = 1:K
                        tmp = l-testtF(c)-k;
                        if tmp>0
                            testy1((c-1)*K+k) = tmp/tau*exp(1-tmp/tau);
                            D_testy1((c-1)*K+k) = (1-tmp/tau)/tau*exp(1-tmp/tau);
                        else
                            testy1((c-1)*K+k) = 0;
                            D_testy1((c-1)*K+k) = 0;
                        end
                    end
                end
                for q = 1:Q
                    if testtH(q)==100
                        testSH(q) = sum(w(:,q).*testy1);
                        if testSH(q)>theta
                            testtH(q) = l;
                            testy(:,q) = testy1;
                            D_testy(:,q) = D_testy1;
                            D_testSH(q) = sum(w(:,q).*D_testy1);
                        end
                    end
                end
                if max(testtH)<100
                    break
                end
            end
            
            %%%%% testing Output Layer %%%%%%%%%%%%%%
            for l = min(testtH):0.2:L
                for q = 1:Q
                    for k = 1:K
                        tmp = l-testtH(q)-k;
                        if tmp>0
                            testY1((q-1)*K+k) = I(q)*tmp/tau*exp(1-tmp/tau);
                            D_testY1((q-1)*K+k) = I(q)*(1-tmp/tau)/tau*exp(1-tmp/tau);
                        else
                            testY1((q-1)*K+k) = 0;
                            D_testY1((q-1)*K+k) = 0;
                        end
                    end
                end
                for p = 1:P
                    if testtO(p)==100
                        testSO(p) = sum(u(:,p).*testY1);
                        if testSO(p)>theta
                            testtO(p) = l;
                            testY(:,p) = testY1;
                            D_testY(:,p) = D_testY1;
                            D_testSO(p) = sum(u(:,p).*D_testY1);
                        end
                    end
                end
                if max(testtO)<100
                    break
                end
            end
            
            %%%%%%%%%% testing err caculate %%%%%%%%%%%%%%%
            testoutput(j,:) = testtO;
            testerr(j) = sum((testtO-testO1(j,:)).^2)/(2*P);
            %%%%%%%%% testing accuracy %%%%%%%%%%
            if testtO<7
                pos = 0;
            else
                pos = 1;
            end
            if pos==testO(j)
                testaccuracy=testaccuracy+1;
            end
        end
        Err(epoch) = sum(err)/J;
        ACC(epoch) = accuracy/J;
        ACC1 = ACC(epoch);
        testErr(epoch) = sum(testerr)/testJ;
        testACC(epoch) = testaccuracy/testJ;
        testACC1 = testACC(epoch);
        [epoch,Err(epoch),ACC(epoch),testErr(epoch),testACC(epoch)];
        if (testACC(epoch)>tmp_testACC && ACC(epoch)>=tmp_ACC)||(testACC(epoch)>=tmp_testACC && ACC(epoch)>tmp_ACC)
            tmp_testACC = testACC(epoch);
            tmp_ACC = ACC(epoch);
            tmp_epoch = epoch;
        end %%% End: the best output
        
        epoch = epoch+1;
    end
    
    kk=[tmp_epoch,Err(tmp_epoch),tmp_ACC,testErr(tmp_epoch),tmp_testACC];
    kkk=kkk+kk;
end
kkk/10
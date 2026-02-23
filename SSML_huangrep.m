%% Description %%
% Date: 08/23/2023
% This is a script that replicates similar results from Huang et. al. (2006)
% using the breast cancer dataset from UCI. Huang et. al. (2006) train a
% support vector classifier using three different weighting schemes:
% unweighted, importance sampling and KMM. Test results are averaged over
% 30 trials. Rather than SVM, this script uses OLS as the learner with the
% same three weighting schemes. 
% The figures from the main paper are created using the following files
%   - Figure 1: "huangrep_1.jpeg" and "huangrep_2.jpeg"
% Prior to running, change FILEPATH in line 18.
% Note: The script does allow for training a support vector classifier but 
% it has been commented out in lines 60, 83-84, 92-93, 99-100, 125-126, 
% 134-135, and 141-142.  

clear
% Replace FILEPATH with location of this script and breast cancer dataset
folder = '/Users/ladanashkina/Library/CloudStorage/OneDrive-EcolePolytechnique/Ecole Polytechnique Courses/Y2-Semester 2/Machine Learning for Causal Inference/Paper/resources/ReplicationFiles/HuangetalReplication';
cd(folder)

% Set Preliminaries
a = 0.5;
b = 0.9;
c = 0.1;
f = 0.45;
g = 0.9;
sigma = .1;
R = 30;
rng(22222)

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 11);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["id", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "y"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
data = readtable("breast-cancer-wisconsin.data", opts);
clear opts

% Clean data
data = rmmissing(data);
[n,k] = size(data);
k = k-2;
data.y(ismember(data.y,2)) = 1;
data.y(ismember(data.y,4)) = 0;

LS_MSE = cell(2,1);
LS_MSE{1} = zeros(R,3);
LS_MSE{2} = zeros(R,3);
%SVM_TE = zeros(R,3);
for r = 1:R
% Create training and test set
train = rand(n,1)>a;
e = rand(n,1);
s = (e>b).*ismember(data.y,1) + (e>c).*ismember(data.y,0);
S = s(train);

X = table2array(data(train,2:1+k));

Xtrn = table2array(data(logical(train.*s),2:1+k));
Xtst = table2array(data(logical(train.*(1-s)),2:1+k));
Xout = table2array(data(logical(1-train),2:1+k));

Ytrn = table2array(data(logical(train.*s),k+2));
Ytst = [];
Yout = table2array(data(logical(1-train),k+2));
[nout,~] = size(Yout);

%% Estimation
% Unweighted
OLS = fitlm(Xtrn,Ytrn);
LS_MSE{1}(r,1) = mean((predict(OLS,Xout)-Yout).^2);
%SVM= fitcsvm(Xtrn,Ytrn);
%SVM_TE(r,1) = mean((predict(SVM,Xout)-Yout).^2);

% Logit weights
Logit = fitglm(X,S,'Distribution', 'binomial','Link','logit');
P = Logit.Fitted.Probability(logical(S));
W = P.^(-1)/sum(P.^(-1));
WLS_L = fitlm(Xtrn,Ytrn,'Weights',W);
LS_MSE{1}(r,2) = mean((predict(WLS_L,Xout)-Yout).^2);
%SVM_L = fitcsvm(Xtrn,Ytrn,'Weights',W);
%SVM_TE(r,2) = mean((predict(SVM_L,Xout)-Yout).^2);

% KMM weights
beta = betaKMM(Xtrn, Ytrn, Xtst, Ytst, sigma, 1, 1);
WLS_KMM = fitlm(Xtrn,Ytrn,'Weights',beta);
LS_MSE{1}(r,3) = mean((predict(WLS_KMM,Xout)-Yout).^2);
%SVM_KMM = fitcsvm(Xtrn,Ytrn,'Weights',beta);
%SVM_TE(r,3) = mean((predict(SVM_KMM,Xout)-Yout).^2);
end

for r = 1:R
%% Create training and test set
train = rand(n,1)>a;
e = rand(n,1);
s = (e>f).*ismember(data.y,1) + (e>g).*ismember(data.y,0);
S = s(train);

X = table2array(data(train,2:1+k));

Xtrn = table2array(data(logical(train.*s),2:1+k));
Xtst = table2array(data(logical(train.*(1-s)),2:1+k));
Xout = table2array(data(logical(1-train),2:1+k));

Ytrn = table2array(data(logical(train.*s),k+2));
Ytst = [];
Yout = table2array(data(logical(1-train),k+2));
[nout,~] = size(Yout);

%% Estimation
% Unweighted
OLS = fitlm(Xtrn,Ytrn);
LS_MSE{2}(r,1) = mean((predict(OLS,Xout)-Yout).^2);
%SVM= fitcsvm(Xtrn,Ytrn);
%SVM_TE(r,1) = mean((predict(SVM,Xout)-Yout).^2);

% Logit weights
Logit = fitglm(X,S,'Distribution', 'binomial','Link','logit');
P = Logit.Fitted.Probability(logical(S));
W = P.^(-1)/sum(P.^(-1));
WLS_L = fitlm(Xtrn,Ytrn,'Weights',W);
LS_MSE{2}(r,2) = mean((predict(WLS_L,Xout)-Yout).^2);
%SVM_L = fitcsvm(Xtrn,Ytrn,'Weights',W);
%SVM_TE(r,2) = mean((predict(SVM_L,Xout)-Yout).^2);

% KMM weights
beta = betaKMM(Xtrn, Ytrn, Xtst, Ytst, sigma, 1, 1);
WLS_KMM = fitlm(Xtrn,Ytrn,'Weights',beta);
LS_MSE{2}(r,3) = mean((predict(WLS_KMM,Xout)-Yout).^2);
%SVM_KMM = fitcsvm(Xtrn,Ytrn,'Weights',beta);
%SVM_TE(r,3) = mean((predict(SVM_KMM,Xout)-Yout).^2);
end

%% Plots
for d=1:2
figure 
pos = [0, 0, 3, 3];
set(gcf,'Units', 'Inches', 'Position', pos)
hold on
    boxplot(LS_MSE{d},'Labels',{'OLS','WLS-Logit','WLS-KMM'},'Colors',(1/255)*[0    114 178],'Symbol','+k','Positions',[.75 2 3.25])
ylim([0.02,.15])
xlim([0,4])
if d == 1
ylabel('MSPE')
end
hold off

set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
graphname = sprintf('huangrep_%i',d);
print(gcf,graphname,'-djpeg')

end
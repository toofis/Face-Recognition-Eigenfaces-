clear
%% 
%Initialisation
N = 112; M = 92; k = 20;

%7 images for train, 3 images for test
tr = 7; tst = 3; 

%Load training images
V = zeros(N*M,k*tr);
folder = 'C:\Users\travellingsalesman\Desktop\ADSP\eigenfaces';

 for i = 1:k
  for j = 1 : tr
      
    face = sprintf('s%d', i);
    ang = sprintf('%d.pgm',j);
    I = fullfile(folder, face, ang);
    I = im2double(imread(I));
    
    V(:, (i-1)*tr+j) = reshape(I,[10304,1]);
  end
 end
 
%%
%Training

%Calculate mean & subtract from columns
m_c = 1/(k*tr) * sum(V,2);
V = bsxfun(@minus, V, m_c);

%Eigenanalysis to (small) covariance matrix
C = V'*V;

[P,S,~] = svd(C);

%Extend dimension
Q = V*P;

%Keep 8 first eigenvalues
d = 8;
Q = Q(:,1:d);

%Eigenspace projection weights
W = Q'*V;

%%
%%Testing

%Load testing images
V_test = zeros(N*M,k*tst);

 for i = 1:k
  for j = 1 : tst
      
    face = sprintf('s%d', i);
    ang = sprintf('%d.pgm',j+7);
    I = fullfile(folder, face, ang);
    I = im2double(imread(I));
    
    V_test(:, (i-1)*tst+j) = reshape(I,[10304,1]);
  end
 end
 
 %Subtract mean
 V_test = bsxfun(@minus, V_test, m_c);
 %Project to eigenspace
 W_test = Q'*V_test;
 
 recognized = zeros(k*tst,1);
 %for each testing image
 for i = 1 : k*tst
     min = inf;
 %for each trained weight
     for j = 1 : k*tr
      %Calculate min distance from training set
      w = norm(W(:,j)-W_test(:,i))^2;
      if w < min
       min = w;
       %If image matches any image from a training face, is categorised as
       %corresponding face
       recognized(i) = ceil(j/tr);
      end   
     end
 end
 

function CrackInspection_final_boost()
%crack inspection
%  The main principle is that the neighbor points in curve have similar tangential direction.
%  The proposed algorithm contains two main steps:track curve self according to tangential direction, 
%  and expand roughly according to normal direction

%  Contributed by Andrew, and written at BJTU, Beijing, China.

clear; clc;
addpath('img\'); 
I=imread('42.bmp');
ptsNum = 9;
if ndims(I) > 2  
    I=rgb2gray(I); 
end
global g_threshold;
g_threshold = 150/255;

I = im2double(I); %normalized to [0,1]
figure; imshow(I,[])
h = imrect; %plot rect
pos = getPosition(h); 
Img = imcrop(I, pos); %crop
imwrite(Img,'D:\HHH.bmp');
return
%figure; plot(1:size(Img,1),Img(:,50));
figure; imshow(Img,[]);
%双边滤波
Img_edge = bilateralFilter(Img, [], 0, 1.0, 1, 0.3);
%引导滤波，计算切线和法线方向(其实也可用双边滤波图像计算)
Img_center = guidedfilter(Img, Img, 2, 0.4^2);  % try r=2, 4, or 8  16  eps=0.1^2, 0.2^2, 0.4^2  0.1^2
%Img_enhancement = abs(Img - Img_center) * 5 + Img_center;
%再次预处理，log归一化，加大对比度
%Img = log_normalization(Img);
Img_edge = log_normalization(Img_edge);
Img_center = log_normalization(Img_center);
[Ix,Iy]=gaussgradient(Img_edge, 1);
Img_gradient = abs(Ix) + abs(Iy);
%figure; imshow(Img_gradient,[]);

%根据Hessian矩阵计算图像中每一个像素点的切线方向uq，和法线方向uf
%uf_row表示图像中的竖直方向，uf_col表示水平方向
[uf_row uf_col uq_row uq_col] = caculate_maxeigenvalue_eigenvector(Img_edge, 2.0);

[t_row t_col] = size(Img_edge);
finalFlag = zeros(size(Img)); %标志位矩阵

%鼠标取点,起点和终点
[seedx, seedy]=ginput(ptsNum);  %the set seed point is very important
for kk=1:(ptsNum-1)
    oriPt(1) = round(seedy(kk));
    oriPt(2) = round(seedx(kk));
    endPt(1) = round(seedy(kk+1));
    endPt(2) = round(seedx(kk+1));
    finalFlag = calcEachLine(oriPt, endPt, finalFlag, Img_edge, Img_gradient, uf_row, uf_col, uq_row, uq_col);
end

finalOutCome = zeros(t_row, t_col, 3);
I1=Img;I2=Img;I3=Img;
I1(finalFlag(:)==1) = 1;
I2(finalFlag(:)==1) = 0;
I3(finalFlag(:)==1) = 1;
I1(finalFlag(:)==2) = 0;
I2(finalFlag(:)==2) = 1;
I3(finalFlag(:)==2) = 0;
I1(finalFlag(:)==3) = 1;
I2(finalFlag(:)==3) = 0;
I3(finalFlag(:)==3) = 0;
finalOutCome(:,:,1)=I1;
finalOutCome(:,:,2)=I2;
finalOutCome(:,:,3)=I3;

figure, imshow(finalOutCome, []);
%imwrite(finalOutCome, 'D:\HH.bmp');
%subplot(2,1,2), imshow(Img, []); %, subplot(2,1,1)

%zImg = imresize(Img, 10, 'bicubic'); %zoom in 'bilinear'
%figure; imshow(zImg,[]);
%imwrite(zImg,'HH.bmp');
end
function [return_finalFlag] = calcEachLine(oriPt, endPt, finalFlag, Img_edge, Img_gradient, uf_row, uf_col, uq_row, uq_col)
%计算起点和终点在图像中的切线方向，定义为主切线方向
global g_threshold;

uq_prim = endPt-oriPt;
ptLen = norm(uq_prim); %L2归一化
uq_prim = uq_prim/ptLen;
uf_prim = [-uq_prim(2) uq_prim(1)]; %主法线方向

%自追踪 首先从终点开始
meanGrayvalue = Img_edge(endPt(1), endPt(2));
[finalFlag meanGrayvalue] = trackCurve(finalFlag, Img_edge, Img_gradient, uf_row, uf_col, uq_row, uq_col, meanGrayvalue, endPt, oriPt, endPt, uq_prim, uf_prim);

meanGrayvalue = Img_edge(oriPt(1), oriPt(2));
[finalFlag meanGrayvalue] = trackCurve(finalFlag, Img_edge, Img_gradient, uf_row, uf_col, uq_row, uq_col, meanGrayvalue, oriPt, oriPt, endPt, uq_prim, uf_prim); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
for ii=1:floor(ptLen)  %沿主切线方向递增点，直到检测完毕
    currentpt = round(oriPt + ii*uq_prim); %当前点
    if ~isnotTrack(finalFlag, uf_prim, currentpt, 10)
        continue;
    end    
    pts_seq = FX_grow(Img_edge, uq_row, uq_col, uf_prim, uq_prim, currentpt, meanGrayvalue, finalFlag);
    if isempty(pts_seq)
        finalFlag = FX_grow_rough(Img_edge, uq_row, uq_col, uf_prim, uq_prim, currentpt, meanGrayvalue, finalFlag);
        continue;
    end
    currentpt = round(mean(pts_seq,1));    
    currentptV = Img_edge(currentpt(1), currentpt(2)); %当前点的灰度值
    %当前点的切线方向
    uq_currentpt = [uq_row(currentpt(1),currentpt(2)) uq_col(currentpt(1),currentpt(2))];
    if isSimilarity(uq_prim, uq_currentpt) && abs(meanGrayvalue-currentptV)<g_threshold 
        %从当前点 自追踪 曲线,返回追踪点pts_seq和标志位矩阵finalFlag
        [finalFlag meanGrayvalue1] = trackCurve(finalFlag, Img_edge, Img_gradient, uf_row, uf_col, uq_row, uq_col, meanGrayvalue, currentpt, oriPt, endPt, uq_prim, uf_prim); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    else
        ind = sub2ind(size(finalFlag), pts_seq(:,1), pts_seq(:,2));
        finalFlag(ind) = 2; %标志位置1
    end
end
return_finalFlag = finalFlag;
end

function [return_finalFlag return_meanV] = trackCurve(finalFlag, Img_edge, Img_gradient, uf_row, uf_col, uq_row, uq_col, meanGrayvalue, currentpt, judge_oriPt, judge_endPt, judge_uq_prim, uf_prim)
%自追踪曲线
global g_threshold;

seed_pt=[];
seed_pt = cat(1, seed_pt, currentpt);
[t_row t_col] = size(Img_edge);
return_meanV = 0;
ncount = 0;
while size(seed_pt,1)>0 % ~isempty(seed_pt)
    pt = seed_pt(end,:);
    seed_pt = seed_pt(1:end-1,:);
    currentptV = Img_edge(pt(1), pt(2));
    if isInline(pt, judge_oriPt, judge_endPt, judge_uq_prim) && isInneighbor(pt, judge_oriPt, judge_endPt, judge_uq_prim) && ...
            (abs(meanGrayvalue-currentptV) < g_threshold)
        ncount = ncount+1;
        return_meanV = return_meanV + currentptV;
    else
        continue;
    end
    uf_currentPt = [uf_row(pt(1),pt(2)) uf_col(pt(1),pt(2))];
    uq_currentPt = [uq_row(pt(1),pt(2)) uq_col(pt(1),pt(2))];

    pts_seq = scanFX(Img_gradient, uf_row, uf_col, uq_row, uq_col, pt, uf_prim);
    ind = sub2ind(size(finalFlag), pts_seq(:,1), pts_seq(:,2));
    finalFlag(ind) = 1;
    pt_avg = mean(pts_seq,1);
    new_pt1 = round(pt_avg + uq_currentPt);
    new_pt2 = round(pt_avg - uq_currentPt);
    
    isinStk = 0;
    if new_pt1(1)>0 && new_pt1(1)<=t_row && new_pt1(2)>0 && new_pt1(2)<=t_col &&  ...
            finalFlag(new_pt1(1),new_pt1(2))==0
        uq_newPt1 = [uq_row(new_pt1(1),new_pt1(2)) uq_col(new_pt1(1),new_pt1(2))];
        
        if isSimilarity(uq_currentPt, uq_newPt1)
            isinStk = 1;
            seed_pt(end+1,:) = new_pt1;
        else
            new_pt = round(pt_avg + 2*uq_currentPt); 
            if new_pt(1)>0 && new_pt(1)<=t_row && new_pt(2)>0 && new_pt(2)<=t_col &&  ...
            finalFlag(new_pt(1),new_pt(2))==0
                uq_newPt = [uq_row(new_pt(1),new_pt(2)) uq_col(new_pt(1),new_pt(2))];
                if isSimilarity(uq_currentPt, uq_newPt)
                    isinStk = 1;
                    seed_pt(end+1,:) = new_pt;
                end
            end
        end
    end
    if new_pt2(1)>0 && new_pt2(1)<=t_row && new_pt2(2)>0 && new_pt2(2)<=t_col &&  ...
        finalFlag(new_pt2(1),new_pt2(2))==0 
        uq_newPt2 = [uq_row(new_pt2(1),new_pt2(2)) uq_col(new_pt2(1),new_pt2(2))];
        if isSimilarity(uq_currentPt, uq_newPt2)
            isinStk = 1;
            seed_pt(end+1,:) = new_pt2;
        else
            new_pt = round(pt_avg - 2*uq_currentPt);
            if new_pt(1)>0 && new_pt(1)<=t_row && new_pt(2)>0 && new_pt(2)<=t_col &&  ...
            finalFlag(new_pt(1),new_pt(2))==0
                uq_newPt = [uq_row(new_pt(1),new_pt(2)) uq_col(new_pt(1),new_pt(2))];
                if isSimilarity(uq_currentPt, uq_newPt)
                    isinStk = 1;
                    seed_pt(end+1,:) = new_pt;
                end
            end
        
        end
    end
    if isinStk == 0
        pts_seq = FX_choose_neighbor(Img_edge, uq_row, uq_col, uf_currentPt, uq_currentPt, new_pt1, meanGrayvalue, finalFlag);
        nPts = size(pts_seq,1);
        if nPts >= 2
            pt_avg = pts_seq(round(nPts/2),:);  %select median value
            seed_pt(end+1,:) = pt_avg;
        end
    end
    
end
return_finalFlag = finalFlag;
return_meanV = return_meanV /ncount;
end

function [pts_seq] = FX_choose_neighbor(Img_edge, uq_row, uq_col, uf_prim, uq_prim, pt, meanGrayvalue, finalFlag)
%在当前点pt的主法线上粗暴扩展
%如果主法线上某点的切线方向与主切线方向一致，选取该点；否则舍弃
global g_threshold;

pts_seq = [];
[t_row t_col] = size(Img_edge);
for ii = -2:2
   new_pt = round(pt + ii*uf_prim);
   if new_pt(1)>0 && new_pt(1)<=t_row && new_pt(2)>0 && new_pt(2)<=t_col
        uq_newPt = [uq_row(new_pt(1),new_pt(2)) uq_col(new_pt(1),new_pt(2))];
        currentptV = Img_edge(new_pt(1), new_pt(2));
        if finalFlag(new_pt(1),new_pt(2)) == 0 && isSimilarity(uq_prim, uq_newPt) && abs(meanGrayvalue-currentptV)<g_threshold
            pts_seq = cat(1, pts_seq, new_pt);
        end
   end
end

end

function [returnFlag] = isInline(currentpt, judge_oriPt, judge_endPt, judge_uq_prim)
uq_oript = currentpt-judge_oriPt;
ptLen = norm(uq_oript)+1e-3; %L2归一化
uq_oript = uq_oript/ptLen;

uq_endpt = judge_endPt-currentpt;
ptLen = norm(uq_endpt)+1e-3; %L2归一化
uq_endpt = uq_endpt/ptLen;

returnFlag = 0;
if uq_oript*judge_uq_prim'>=0 && uq_endpt*judge_uq_prim'>=0
    returnFlag = 1;
end
end

function [returnFlag] = isInneighbor(currentpt, judge_oriPt, judge_endPt, judge_uq_prim)
uq_pt = currentpt-judge_oriPt;
ptLen = norm(uq_pt)+1e-3; %L2归一化
uq_pt = uq_pt/ptLen;

msim = uq_pt*judge_uq_prim';
normLen = ptLen * sqrt(1-msim*msim);
returnFlag = 1;
if normLen > 30
    returnFlag = 0;
end
end

function [returnFlag] = isnotTrack(finalFlag, uf_prim, pt, thr)
[t_row t_col] = size(finalFlag);
returnFlag = 1;
for ii=0:thr
   new_pt = round(pt + ii*uf_prim);
   if new_pt(1)>0 && new_pt(1)<=t_row && new_pt(2)>0 && new_pt(2)<=t_col
        if finalFlag(new_pt(1), new_pt(2)) ~= 0
            returnFlag = 0;
            break;
        end
   else
       break;
   end
end
if returnFlag == 0
    return;
end
for ii=1:thr
   new_pt = round(pt - ii*uf_prim);
   if new_pt(1)>0 && new_pt(1)<=t_row && new_pt(2)>0 && new_pt(2)<=t_col
        if finalFlag(new_pt(1), new_pt(2)) ~= 0
            returnFlag = 0;
            break;
        end
   else
       break;
   end
end
end

function [return_finalFlag] = FX_grow_rough(Img_edge, uq_row, uq_col, uf_prim, uq_prim, pt, meanGrayvalue, finalFlag)
%在当前点pt的主法线上粗暴扩展
%如果主法线上某点的切线方向与主切线方向一致，选取该点；否则舍弃
global g_threshold;

pts_seq = [];
[t_row t_col] = size(Img_edge);
for ii=0:2
   new_pt = round(pt + ii*uf_prim);
   if new_pt(1)>0 && new_pt(1)<=t_row && new_pt(2)>0 && new_pt(2)<=t_col
        uq_newPt = [uq_row(new_pt(1),new_pt(2)) uq_col(new_pt(1),new_pt(2))];
        currentptV = Img_edge(new_pt(1), new_pt(2));
        if (isSimilarity(uq_prim, uq_newPt)) && abs(meanGrayvalue-currentptV)<g_threshold
            pts_seq = cat(1, pts_seq, new_pt);
            finalFlag(new_pt(1), new_pt(2)) = 3;
        end
   else
       break;
   end
end
for ii=1:2
   new_pt = round(pt - ii*uf_prim);
   if new_pt(1)>0 && new_pt(1)<=t_row && new_pt(2)>0 && new_pt(2)<=t_col
        uq_newPt = [uq_row(new_pt(1),new_pt(2)) uq_col(new_pt(1),new_pt(2))];
        currentptV = Img_edge(new_pt(1), new_pt(2));
        if (isSimilarity(uq_prim, uq_newPt)) && abs(meanGrayvalue-currentptV)<g_threshold
            pts_seq = cat(1, pts_seq, new_pt);
            finalFlag(new_pt(1), new_pt(2)) = 3;
        end
   else
       break;
   end
end
return_finalFlag = finalFlag;
end

function [pts_seq] = FX_grow(Img_edge, uq_row, uq_col, uf_prim, uq_prim, pt, meanGrayvalue, finalFlag)
%在当前点pt的主法线上粗暴扩展
%如果主法线上某点的切线方向与主切线方向一致，选取该点；否则舍弃
global g_threshold;

pts_seq = [];
[t_row t_col] = size(Img_edge);
for ii=0:30
   new_pt = round(pt + ii*uf_prim);
   if new_pt(1)>0 && new_pt(1)<=t_row && new_pt(2)>0 && new_pt(2)<=t_col
        uq_newPt = [uq_row(new_pt(1),new_pt(2)) uq_col(new_pt(1),new_pt(2))];
        currentptV = Img_edge(new_pt(1), new_pt(2));
        if (~isnotTrack(finalFlag, uq_prim, new_pt, 10)) && isSimilarity(uq_prim, uq_newPt) && abs(meanGrayvalue-currentptV)<g_threshold
            pts_seq = cat(1, pts_seq, new_pt);
        end
   else
       break;
   end
end
for ii=1:30
   new_pt = round(pt - ii*uf_prim);
   if new_pt(1)>0 && new_pt(1)<=t_row && new_pt(2)>0 && new_pt(2)<=t_col
        uq_newPt = [uq_row(new_pt(1),new_pt(2)) uq_col(new_pt(1),new_pt(2))];
        currentptV = Img_edge(new_pt(1), new_pt(2));
        if (~isnotTrack(finalFlag, uq_prim, new_pt, 10)) && isSimilarity(uq_prim, uq_newPt) && abs(meanGrayvalue-currentptV)<g_threshold
            pts_seq = cat(1, pts_seq, new_pt);
        end
   else
       break;
   end
end

end

function [pts_seq] = scanFX(Img_gradient, uf_row, uf_col, uq_row, uq_col, pt, uf_prim)
%在当前点的法线上扫描，根据灰度，坐标，梯度，切线方向;
%注意这里是当前点的法线上扫描边界点，然后选取两个边界点之间的所有点,组成要检测的线
uf_currentPt = [uf_row(pt(1),pt(2)) uf_col(pt(1),pt(2))];
uq_currentPt = [uq_row(pt(1),pt(2)) uq_col(pt(1),pt(2))];
pts_seq = [];
pts_seq = cat(1, pts_seq, pt);

[t_row t_col] = size(Img_gradient);
myT = Img_gradient(pt(1),pt(2));
for ii=1:10
   new_pt = round(pt + ii*uf_prim);
   if new_pt(1)>0 && new_pt(1)<=t_row && new_pt(2)>0 && new_pt(2)<=t_col
        uq_newPt = [uq_row(new_pt(1),new_pt(2)) uq_col(new_pt(1),new_pt(2))];
        msim = abs(uq_currentPt*uq_newPt');
        T = msim * Img_gradient(new_pt(1),new_pt(2));
        myT(end+1) = T;
   else
       break;
   end
end
[maxV maxInd] = max(myT);
for ii=1:maxInd-1
    new_pt = round(pt + ii*uf_prim);
    pts_seq = cat(1, pts_seq, new_pt);
end

myT = Img_gradient(pt(1),pt(2));
for ii=1:10
   new_pt = round(pt - ii*uf_prim);
   if new_pt(1)>0 && new_pt(1)<=t_row && new_pt(2)>0 && new_pt(2)<=t_col
        uq_newPt = [uq_row(new_pt(1),new_pt(2)) uq_col(new_pt(1),new_pt(2))];
        msim = abs(uq_currentPt*uq_newPt');
        T = msim * Img_gradient(new_pt(1),new_pt(2));
        myT(end+1) = T;
   else
       break;
   end
end
[maxV maxInd] = max(myT);
for ii=1:maxInd-1
    new_pt = round(pt - ii*uf_prim);
    pts_seq = cat(1, pts_seq, new_pt);
end

end

function [returnFlag] = isSimilarity(uq_currentPt, uq_newPt)
msim = uq_currentPt * uq_newPt';
if abs(msim) >= 0.9
    returnFlag = 1;
else
    returnFlag = 0;
end
end

function [returnImg] = log_normalization(Img)
Img = Img*255;
logII = log((Img+1).*0.5);      %compress image data to eliminate outliers
meanlogII = mean(logII(:));
logII = (logII - meanlogII) ./ var(logII(:));%(logII + meanlogII);
minA=min(min(logII));maxA=max(max(logII));
returnImg = (logII-minA)/(maxA-minA)*255;
returnImg = returnImg/255;
end

function [uf_row uf_col uq_row uq_col] = caculate_maxeigenvalue_eigenvector(Image, s)
[Ix,Iy]=gaussgradient(Image, s);
[Ixx,Ixy]=gaussgradient(Ix,s);
[Ixy,Iyy]=gaussgradient(Iy,s);

% Calculate max eigenvalues (lamdam) of Hessian matrix [Ixx Ixy; Ixy Iyy]
a=Ixx;b=Ixy;c=Iyy;
q=0.5*((a+c)+sign(a+c).*sqrt((a-c).^2+4*b.^2));
lamda1=q;
lamda2=(a.*c-b.^2)./q;
lamdam=max(lamda1,lamda2);

%最大特征值对应的特征向量为法线方向
uf_col = abs(b)./sqrt(b.^2+(lamdam-a).^2);
uf_row = sign(b).*(lamdam-a)./sqrt(b.^2+(lamdam-a).^2);
%计算切线方向
uq_row = uf_col;
uq_col = -uf_row;
end

function [gx,gy]=gaussgradient(IM,sigma)
%GAUSSGRADIENT Gradient using first order derivative of Gaussian.
%  [gx,gy]=gaussgradient(IM,sigma) outputs the gradient image gx and gy of
%  image IM using a 2-D Gaussian kernel. Sigma is the standard deviation of
%  this kernel along both directions.
%
%  Contributed by Guanglei Xiong (xgl99@mails.tsinghua.edu.cn)
%  at Tsinghua University, Beijing, China.

%determine the appropriate size of kernel. The smaller epsilon, the larger
%size.
epsilon=1e-2;
halfsize=ceil(sigma*sqrt(-2*log(sqrt(2*pi)*sigma*epsilon)));
size=2*halfsize+1;
%generate a 2-D Gaussian kernel along x direction
for i=1:size
    for j=1:size
        u=[i-halfsize-1 j-halfsize-1];
        hx(i,j)=gauss(u(1),sigma)*dgauss(u(2),sigma);
    end
end
hx=hx/sqrt(sum(sum(abs(hx).*abs(hx))));
%generate a 2-D Gaussian kernel along y direction
hy=hx';
% surf(hx)
%2-D filtering
gx=imfilter(IM,hx,'replicate','conv');
gy=imfilter(IM,hy,'replicate','conv');
end

function y = gauss(x,sigma)
%Gaussian
y = exp(-x^2/(2*sigma^2)) / (sigma*sqrt(2*pi));
end

function y = dgauss(x,sigma)
%first order derivative of Gaussian
y = -x * gauss(x,sigma) / sigma^2;
end
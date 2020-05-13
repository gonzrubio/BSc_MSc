%% demo_hankel_levels.m
%
%  Gonzalo Rubio July, 2018
%  University of Alberta, SAIG & Global Seismology Group

%% demo level-2
% Works. output is same as input, does same as original code

clf ; clear all; clc ; close all ;

% Mine
data = randi([1 10],7,7,8) ; % data(t,x,y)
[nt,nx,ny] = size(data);
DATA_FX_f_mine = zeros(nt,nx,ny) ;

% Ncol = floor(ny/2)+1;
% Nrow = ny-Ncol+1;
% Lcol = floor(nx/2)+1;
% Lrow = nx-Lcol+1;
% 
% %for j = 1:nt
% for j = 1
%     M = 0 ;
%     for lc = 1:Lcol
%         for lr = 1:Lrow
%             tmp_fx(lr+lc-1,:)  = squeeze(data(j,lr+lc-1,:)) ;
%             for ic = 1:Ncol
%                 for ir = 1:Nrow
%                     M(((lr*Nrow)-Nrow+ir),((lc*Ncol)-Ncol+ic)) = tmp_fx(lr+lc-1,ir+ic-1) ;
%                end
%             end
%         end
%     end
% 
%     % Average
%     tic;
%     Count = zeros(ny,nx);
%     tmp = zeros(ny,nx);
%     tmp2 = zeros(ny,nx);
% 
%     for lc = 1:Lcol
%      for lr = 1:Lrow
%       for ic = 1:Ncol
%        for ir = 1:Nrow
%            Count(ir+ic-1,lr+lc-1) = Count(ir+ic-1,lr+lc-1)+1;
%            tmp(ir+ic-1,lr+lc-1)  = tmp(ir+ic-1,lr+lc-1) + M(((lr*Nrow)-Nrow+ir),((lc*Ncol)-Ncol+ic));
%        end
%       end
%      end
%     end
%     DATA_FX_f_mine(j,:,:) = (tmp./Count).' ;
%     time1=toc;
% end

% %Original
% DATA_FX_tmp = data ; 
% 
% % Size of the Hankel Matrix for y.
% 
% Ncol = floor(ny/2)+1;
% Nrow = ny-Ncol+1;
% 
% % Size of Block Hankel Matrix with hankel blocks of Hankel Matrices in x.
% 
% Lcol = floor(nx/2)+1;
% Lrow = nx-Lcol+1;
% H = zeros(Nrow*Lrow,Ncol*Lcol) ;
% 
% DATA_FX_f_og = zeros(nt,nx,ny);
% %for j= 1:nt
% for j = 1
%     M = 0;
%     for lc = 1:Lcol
%         for lr = 1:Lrow
%             tmp_fx(lr+lc-1,:)  = squeeze(DATA_FX_tmp(j,lr+lc-1,:)).' ;
%             for ic = 1:Ncol
%                 for ir = 1:Nrow
%                     M(((lr*Nrow)-Nrow+ir),((lc*Ncol)-Ncol+ic)) = tmp_fx(lr+lc-1,ir+ic-1) ;
%                 end
%             end
%         end
%     end
%     
%     tic
%     Count = zeros(ny,nx);
%     tmp = zeros(ny,nx);
%     tmp2 = zeros(ny,nx);
% 
%     for lc = 1:Lcol
%      for lr = 1:Lrow
%       for ic = 1:Ncol;
%        for ir = 1:Nrow;
%            Count(ir+ic-1,lr+lc-1) = Count(ir+ic-1,lr+lc-1)+1;
%            tmp(ir+ic-1,lr+lc-1)  = tmp(ir+ic-1,lr+lc-1) + M(((lr*Nrow)-Nrow+ir),((lc*Ncol)-Ncol+ic));
%        end;
%       end
%       tmp2(:,lr+lc-1) = tmp(:,lr+lc-1)./Count(:,lr+lc-1);
%       DATA_FX_f_og(j,lr+lc-1,:) = tmp2(:,lr+lc-1).';
% 
%      end
%     end
%     time2=toc;
% end

% Ncol = floor(ny/2);
% Nrow = ny-Ncol;
% Lcol = floor(nx/2);
% Lrow = nx-Lcol;
% 
% 
% for j = 1
%     M = 0 ;
%     for lc = 1:Lcol
%         for lr = 1:Lrow
%             tmp_fx(lr+lc-1,:)  = squeeze(data(j,lr+lc-1,:)) ;
%             for ic = 1:Ncol
%                 for ir = 1:Nrow
%                     M(((lr*Nrow)-Nrow+ir),((lc*Ncol)-Ncol+ic)) = tmp_fx(lr+lc-1,ir+ic-1) ;
%                end
%             end
%         end
%     end
% 
%     % Average
%     tic;
%     C=Ave2DT(M,Ncol,Nrow,Lcol,Lrow);
%     time1=toc;
% end
%     
%% Demo level-3
% Output is the same as input and orientation is the same
 
clf ; clear all; clc ; close all ;

% data(t,x1,x2,x3)
din = randi([1 10],5,4,4,7) ; 
[nt,nx1,nx2,nx3] = size(din) ;
dout = zeros(nt,nx1,nx2,nx3) ;

% Size of Hankel Matrices in x3
x3col = floor(nx3/2)+1 ;
x3row = nx3-x3col+1 ;

% Size of the Hankel Matrix for x2
x2col = floor(nx2/2)+1 ;
x2row = nx2-x2col+1 ;

% Size of the Hankel Block Hankel Matrix
x1col = floor(nx1/2)+1 ;
x1row = nx1-x1col+1 ;

tic
for j = 1:nt
    % Form level-3 HBBHHB matrix 
    
    data = squeeze(din(j,:,:,:)) ;
    M = zeros(x3row*x2row*x1row,x3col*x2col*x1col) ;
    for x1c = 1:x1col
        for x1r = 1:x1row
           % Grab x2x3-plane from x1x2x3-data
           tmp_fx2x3 = squeeze(data(x1r+x1c-1,:,:)) ;            % data(x1,x2,x3) -> data(x2,x3)
            for x2c = 1:x2col
                for x2r = 1:x2row
                    % Grab x3-vector from xy-data
                    tmp_fx3 = squeeze(tmp_fx2x3(x2c+x2r-1,:)) ;  % data(x2,x3) -> data(x3)
                    for x3c = 1:x3col
                        for x3r = 1:x3row
                            ridx = (x1r-1)*x2row*x3row+(x2r-1)*x3row+x3r ;
                            cidx = (x1c-1)*x2col*x3col+(x2c-1)*x3col+x3c ;
                            M(ridx,cidx) = tmp_fx3(x3r+x3c-1) ;
                        end
                    end     
                end
            end
        end
    end


    % Sum along anti-diagonals to recover signal 

    Count = zeros(nx1,nx2,nx3) ;
    tmp = zeros(nx1,nx2,nx3) ;
    for x1c = 1:x1col
        for x1r = 1:x1row
            for x2c = 1:x2col
                for x2r = 1:x2row
                    for x3c = 1:x3col
                        for x3r = 1:x3row
                            Count(x1c+x1r-1,x2r+x2c-1,x3r+x3c-1) = Count(x1c+x1r-1,x2r+x2c-1,x3r+x3c-1)+1 ;
                            ridx = (x1r-1)*x2row*x3row+(x2r-1)*x3row+x3r ;
                            cidx = (x1c-1)*x2col*x3col+(x2c-1)*x3col+x3c ;                           
                            tmp(x1c+x1r-1,x2r+x2c-1,x3r+x3c-1) = tmp(x1c+x1r-1,x2r+x2c-1,x3r+x3c-1) + M(ridx,cidx) ;                        
                        end
                    end
                end
            end
        end
    end
    dout(j,:,:,:) = tmp./Count ;

end
toc

%% Demo level-4
% Output is the same as input and orientation is the same

% clf ; clear all; clc ; close all ;
% 
% % data(t,x1,x2,x3,x4)
% din = randi([1 10],5,3,3,3,3) ;     
% [nt,nx1,nx2,nx3,nx4] = size(din) ;
% dout = zeros(nt,nx1,nx2,nx3,nx4) ;
% 
% % Size of the Hankel Matrix for x4
% x4col = floor(nx4/2)+1 ;
% x4row = nx4-x4col+1 ;
% 
% % Size of Block Hankel Matrix of hankel blocks in x3
% x3col = floor(nx3/2)+1 ;
% x3row = nx3-x3col+1 ;
% 
% % Size of Hankel Block of Block Hankel Matrices of hankel blocks in x2
% x2col = floor(nx2/2)+1 ;
% x2row = nx2-x2col+1 ;
% 
% % Size of Block Hankel of Hankel Blocks of Block Hankel Matrices of hankel blocks in x1
% x1col = floor(nx1/2)+1 ;
% x1row = nx1-x1col+1 ;
% 
% for j = 1:nt
%     % Form level-4 BHHBBHHB matrix
%     
%     data = squeeze(din(j,:,:,:,:)) ;
%     M = zeros(x4row*x3row*x2row*x1row,x4col*x3col*x2col*x1col) ;
%     for x1c = 1:x1col
%         for x1r = 1:x1row
%             % Grab x2x3x4-volume from x1x2x3x4-data
%             tmp_fx2x3x4 = squeeze(data(x1r+x1c-1,:,:,:)) ;              % data(x1,x2,x3,x4) -> data(x2,x3,x4)
%             for x2c = 1:x2col
%                 for x2r = 1:x2row
%                    % Grab x3x4-plane from x2x3x4-data
%                    tmp_fx3x4 = squeeze(tmp_fx2x3x4(x2r+x2c-1,:,:)) ;    % data(x2,x3,x4) -> data(x3,x4)
%                     for x3c = 1:x3col
%                         for x3r = 1:x3row
%                             % Grab x4-vector from x3x4-data
%                             tmp_fx4 = squeeze(tmp_fx3x4(x3r+x3c-1,:)) ; % data(x3,x4) -> data(x4)
%                             for x4c = 1:x4col
%                                 for x4r = 1:x4row
%                                     ridx = x4r+(x1r-1)*x2row*x3row*x4row+(x2r-1)*x3row*x4row+(x3r-1)*x4row ;
%                                     cidx = x4c+(x1c-1)*x2col*x3col*x4col+(x2c-1)*x3col*x4col+(x3c-1)*x4col ;
%                                     M(ridx,cidx) = tmp_fx4(x4r+x4c-1) ;
%                                 end
%                             end     
%                         end
%                     end
%                 end
%             end
%         end
%     end
% 
% 
%     % Sum along anti-diagonals to recover signal 
% 
%     Count = zeros(nx1,nx2,nx3,nx4) ;
%     tmp = zeros(nx1,nx2,nx3,nx4) ;
%     for x1c = 1:x1col
%         for x1r = 1:x1row
%             for x2c = 1:x2col
%                 for x2r = 1:x2row
%                     for x3c = 1:x3col
%                         for x3r = 1:x3row
%                             for x4c = 1:x4col
%                                 for x4r = 1:x4row
%                                     Count(x1c+x1r-1,x2r+x2c-1,x3r+x3c-1,x4r+x4c-1) = Count(x1c+x1r-1,x2r+x2c-1,x3r+x3c-1,x4r+x4c-1)+1 ;
%                                     ridx = x4r+(x1r-1)*x2row*x3row*x4row+(x2r-1)*x3row*x4row+(x3r-1)*x4row ;
%                                     cidx = x4c+(x1c-1)*x2col*x3col*x4col+(x2c-1)*x3col*x4col+(x3c-1)*x4col ;
%                                     tmp(x1c+x1r-1,x2r+x2c-1,x3r+x3c-1,x4r+x4c-1) = tmp(x1c+x1r-1,x2r+x2c-1,x3r+x3c-1,x4r+x4c-1) + M(ridx,cidx) ;                        
%                                 end
%                             end
%                         end
%                     end
%                 end
%             end
%         end
%     end
%     dout(j,:,:,:,:) = tmp./Count ;
%     
% end
   




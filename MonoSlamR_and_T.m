%% 1. Synthetic scene
clear; close all;

% ------------- Camera intrinsics --------------------------
fx = 800; fy = 800; cx = 320; cy = 240;
K = cameraIntrinsics([fx fy],[cx cy],[640 480]); % 640x480 image

% ------------- 3‑D points ---------------------------------------------------------
N = 200;
Xw = [rand(N,1)*10-5, ... % X in [-5,5]
      rand(N,1)*10-5, ... % Y in [-5,5]
      rand(N,1)*10+5];    % Z in [5,15]

% ------------- Camera 1: origin -----------------------------------------------
Tcw1 = rigidtform3d;          % rotation is identity, translation is zero
R1 = Tcw1.Rotation;
t1 = Tcw1.Translation;        % 1x3 row vector

% ------------- Camera 2: rotation + translation -----------------------------------------
theta = deg2rad(10);
R2 = axang2rotm([0 0 1 theta]); % rotate about Z axis 15°
% t2 = [0.8 0.3 1.6]; % 1x3 row vector
t2 = rand(1,3)*3;

% ------------- Projection function ----------------------------------------------
proj = @(R,t,point3d) ProjectPoints_(point3d, R, t, fx, fy, cx, cy);

% ------------- Generate pixel coordinates (no noise) -----------------------------------
pointPixe1 = proj(R1,t1,Xw);
pointPixe2 = proj(R2,t2,Xw);

% ------------- Add noise ---------------------------------------------
sigmaPix = 1;       % 1 px Gaussian noise
matchPoint1n = pointPixe1 + sigmaPix*randn(size(pointPixe1));
matchPoint2n = pointPixe2 + sigmaPix*randn(size(pointPixe2));

%% 2. Feature matching (using all point pairs directly)

%% 3. Estimate essential matrix
[E,inliersIdx] = estimateEssentialMatrix(matchPoint1n, matchPoint2n, K);

pointPixe1_in = matchPoint1n(inliersIdx,:);
pointPixe2_in = matchPoint2n(inliersIdx,:);

%% 4. Recover relative pose
% [relativePose,~] = estrelpose(E,K,pts1_in,pts2_in);

% using SVD to get the rotation R and translation t
[U,Sig,V] = svd(E);

% Ensure that the subsequently obtained R still belongs to SO(3)
if det(U) < 0
    U = -U;
end
if det(V) < 0
    V = -V;
end
W = [0 -1 0; 1 0 0; 0 0 1];
Ra = U*W*V';
Rb = U*W'*V';
Ta = U(:,3);
Tb = -Ta;

cameraProjection1 = K.K*[eye(3) zeros(3,1)];

% get the four projection matrices for camera 2
cameraProj2 = zeros(3,4,4);
cameraProj2(:,:,1) = [Ra Ta];
cameraProj2(:,:,2) = [Ra Tb];
cameraProj2(:,:,3) = [Rb Ta];
cameraProj2(:,:,4) = [Rb Tb];
point3D_tri4 = zeros(3,4);

for s = 1:4
    point3D_tri4(:,s) = triangulate(pointPixe1_in(1,:), pointPixe2_in(1,:), cameraProjection1, K.K*cameraProj2(:,:,s));
    if point3D_tri4(3,s) > 0
        ss = cameraProj2(:,:,s)*[point3D_tri4(:,s);1];
        if ss(3) > 0
            break;
        end
    end
end
cameraProjection2 = K.K*cameraProj2(:,:,s);

%% 5. Triangulation --------------------------------------------------------
point3D_tri = triangulate(pointPixe1_in, pointPixe2_in, cameraProjection1, cameraProjection2);

% compute error
XwInliers = Xw(inliersIdx,:);
scale = point3D_tri(1,3)/XwInliers(1,3);
point3D_tri = point3D_tri/scale;
err_tri = sqrt(mean(sum((point3D_tri - XwInliers).^2,2)));

%% -------------------------------------------------------------------------
%% 6. Result visualization --------------------------------------------------------
%% -------------------------------------------------------------------------
figure('Name','Triangulation Demo','NumberTitle','off');
scatter3(Xw(:,1),Xw(:,2),Xw(:,3),'bx','LineWidth',1.5); hold on;
scatter3(point3D_tri(:,1),point3D_tri(:,2),point3D_tri(:,3),'ro','LineWidth',1.5);
legend('Ground Truth','Reconstructed','Location','best');
grid on; axis equal; xlabel('X'); ylabel('Y'); zlabel('Z');
title(sprintf('RMSE = %.4f m',err_tri));

%% -------------------------------------------------------------------------
%% 7. Custom projection function (use only when Toolbox not available) --------------------
%% -------------------------------------------------------------------------
function uv = ProjectPoints_(worldPoints, R, t, fx, fy, cx, cy)
    % Project 3-D points to pixel coordinates
    % Input:
    %   worldPoints : Nx3 matrix
    %   R : 3x3 rotation matrix
    %   t : 1x3 translation vector
    %   fx,fy : focal lengths
    %   cx,cy : principal point coordinates
    %
    % Output:
    %   uv : Nx2 pixel coordinates

    % Camera coordinates
    Xc = (R*worldPoints'  + t')'; % Nx3

    % Projection (perspective projection)
    u = (fx * Xc(:,1) + cx * Xc(:,3)) ./ Xc(:,3);
    v = (fy * Xc(:,2) + cy * Xc(:,3)) ./ Xc(:,3);
    uv = [u, v];
end

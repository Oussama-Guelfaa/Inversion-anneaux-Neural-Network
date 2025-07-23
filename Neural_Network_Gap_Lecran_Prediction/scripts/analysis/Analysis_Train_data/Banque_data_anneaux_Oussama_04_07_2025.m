% Anneaux Long pour Oussama


% Ce prog crée une banque de données d'anneaux à partir du champ calculé avec substrat

% On fixe n et r
% On fait varier le gap et la distance d'observation (L_ecran_subs)

% Utilise les datas de 
% load('profile_exp_PS_3um_z_positive.mat','n_m','n_p','r_p')


clear all
close all

%stop

% constantes
global epsilon0 mu0
c_cst=2.99792458e8;
epsilon0=8.854187817e-12;
mu0=4*pi*1e-7;
Zo=sqrt(mu0/epsilon0);



calib=0.0883e-6;
alpha_scat=0.92 % fitté à la main
npts=1000;


lbda=0.532
k0=2*pi/lbda

load('profile_exp_PS_3um_z_positive.mat','n_m','n_p','r_p','I_profiles','r_exp')
n1=n_m
n_int=n_p%1.5615
D_part=2*r_p*1e6

% d'après test de convergence ilfaut prendre degre=30
degre=30
%degre=7

I_zcut2=I_profiles(50,:);
figure;plot(r_exp*1e6,I_zcut2)

% % indices
% n1=1.33 % EAU
% %n2=1.4348
% % nouvelle valeur au 17 09 donnée par Nicolas Fares
% n2=1.52 % LAME DE VERRE  refractive index info sur SiO2 quartz
% 
% % Il faut prendre lame de verre sodocalcique
% 
% % indice bille
% n_int=1.587
% D_part=2*1.516

% Il faut prendre lame de verre sodocalcique
n2=1.5261 % Mail N. Fares 04/02/2025 


% % d'après test de convergence ilfaut prendre degre=30
% degre=30

%%%% Nouvelles valeurs cf mail Fares 21/11/2024
%n_int=1.5792
%D_part=2*1.477




Rmax=13*lbda; % doit etre le meme que dans le prog Main_...
%Rmax=20*lbda; % doit etre le meme que dans le prog Main_...


%%%%%%%%  Tracé des données de Manip
%load MieExperiment_surface.mat
%Rexp=MieExp(:,1);Iexp=MieExp(:,2); 
% Zcut en microns
%Ncut=1000
%Z_cut=linspace(10.15,11.35,Ncut)
%L_ecran_subs = 15  % distance interface eau/verre  - plan focal image
%gap_sphere_vect=[0.015 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.225 0.25 0.275 0.3 0.325 0.35 0.375 0.4 0.425 0.45 0.475 0.5 0.525 0.55 0.575 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1 1.1 1.2 1.3]

gap_sphere_vect=[0.005:0.005:0.7]%linspace(0.005,1.5,200) 

%gap_sphere_vect=[2.005:0.005:3.]%linspace(0.005,1.5,200) 

L_ecran_subs_vect=[8:0.025:12]

%%%%%%%%%%%%%%%%%%%% Parametres de Mie ds l'espace libre
n_ext=n1

nbille=n_int%+i*1e-7;
nb=n_ext; eps_b=nb^2;
nn=[nb,nbille]; 
    n_strates=nb;z_strates=[];
    nm=nbille
    Centre=[0,0,0];n_particule={nm};r_particule={D_part/2};
   % degre=20; % degre de polarisabilité en YLM
% Il faut mettre 30

 [init,Centre_nv,n_strates_nv,z_strates_nv]=retcarminati(n_strates,z_strates,Centre,n_particule,r_particule,k0,degre);% objet test
       C_inc=[pi,pi/2];
%       S_inc=[exp(-i*k0*nb*1e6*(D_part/2+(D_part/2+gap))),0]
       S_inc=[1,0];
       [a,Pt_incident,Flux_Diffracte,Pertes_particule,Flux_Diffuse_particule]=retcarminati(init,C_inc,S_inc,struct('clone',1));

% ========================================================================
%  Dataset generation for Neural Network training
%  It automatically:
%     1. Creates/cleans the output folder "dataset"
%     2. Saves each simulated intensity profile as a PNG figure
%     3. Stores the corresponding raw data (r, I) in a MAT‑file
%     4. Updates a metadata table (gap, L) → labels.csv + labels.mat
%  Author: Oussama Guelfaa – 2025‑06‑03
%  -----------------------------------------------------------------------

% Section ajouté par Oussama----------------------------------------------------
outDir = fullfile(pwd,'dataset');
if ~exist(outDir,'dir')
    mkdir(outDir);
end

nSamples = numel(gap_sphere_vect)*numel(L_ecran_subs_vect);
fileNames     = strings(nSamples,1);
gap_values_um = zeros(nSamples,1);
L_values_um   = zeros(nSamples,1);
sampleIdx = 1;



%%%%%%%%% Calcul anneaux avec substrat
% Boucle sur les différentes hauteurs entre le bas de la sphère et l'interface
  for qq=1:length(gap_sphere_vect) 
      qq
tic;

             gap=gap_sphere_vect(qq)                  ;

% Boucle sur les différents Zcut : position plan de coupe par rapport à l'interface


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CALCUL DES ANNEAUX AVEC LE SUBSTRAT par RETCARM...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      d_interface_bille=gap;
      n_strates_SUBS=[n2,n1];z_strates_SUBS=[D_part/2+d_interface_bille]; % Strates de haut en bas origine : le centre de la sphere
      nb_SUBS=n_strates_SUBS(2);
      nm=n_int;
      Centre=[0,0,0];n_particule={nm};r_particule={D_part/2};
 
[init_SUBS,Centre_nv_SUBS,n_strate_nv_SUBS,z_strate_nv_SUBS]=retcarminati(n_strates_SUBS,z_strates_SUBS,Centre,n_particule,r_particule,k0,degre);% objet test
% Voir help retcarminati faisceau vient du bas
 C_inc_SUBS=[0,pi/2];
 S_inc_SUBS=[1,0];
 S_inc_SUBS=S_inc_SUBS/norm(S_inc_SUBS);
 [a_SUBS,Pt_incident_SUBS,Flux_Diffracte_SUBS,Pertes_SUBS,Flux_Diffuse_SUBS]=retcarminati(init_SUBS,C_inc_SUBS,S_inc_SUBS);
t1(qq)=toc

for pp=1:length(L_ecran_subs_vect)
tic;

 L_ecran_subs=L_ecran_subs_vect(pp);
 Zcut_substrat(qq,pp)=L_ecran_subs+D_part/2+d_interface_bille;
 x=linspace(0,Rmax,npts);y=0;z=Zcut_substrat(qq,pp);

 [etot_subs,otot]=retcarminati(init_SUBS,a_SUBS,{x,y,z},struct('clone',1,'champ_total',1)); %total
 [einc_subs,oinc]=retcarminati(init_SUBS,a_SUBS,{x,y,z},struct('clone',1,'champ_total',2)); %total

eeinc_subs=squeeze(einc_subs);Exinc_subs=eeinc_subs(:,1);
  
eetot_subs=squeeze(etot_subs);Extot_subs=eetot_subs(:,1);Eytot_subs=eetot_subs(:,2);Eztot_subs=eetot_subs(:,3);     
         
I_subs(pp,qq,:)=abs(Extot_subs).^2;
I_subs_inc(pp,qq,:)=abs(Exinc_subs).^2;

%     figure(pp);hold on;
%        plot(x,abs(Extot_subs./Exinc_subs).^2,'k-','LineWidth',3);hold on;plot(r_exp*1e6,I_zcut2)
    t2(qq,pp)=toc  

%stop

%%%%% Champ dans l'autre direction (y)
% y=linspace(0,Rmax,npts);x=0;z=Zcut_substrat(qq,pp);
% 
%  [etot_subs2,otot2]=retcarminati(init_SUBS,a_SUBS,{x,y,z},struct('clone',1,'champ_total',1)); %total
%  [einc_subs2,oinc2]=retcarminati(init_SUBS,a_SUBS,{x,y,z},struct('clone',1,'champ_total',2)); %total
% 
%   eeinc_subs2=squeeze(einc_subs2);Exinc_subs2=eeinc_subs2(:,1);
% 
%   eetot_subs2=squeeze(etot_subs2);Extot_subs2=eetot_subs2(:,1);Eytot_subs2=eetot_subs2(:,2);Eztot_subs2=eetot_subs2(:,3);     
%          
% I_subs2(pp,qq,:)=abs(Extot_subs2).^2;
% I_subs2_inc(pp,qq,:)=abs(Exinc_subs2).^2;
% 
%      figure(10*qq);hold on;
%         plot(y,abs(Extot_subs2).^2,'k-+','LineWidth',3);
% stop

% Section ajouté par Oussama----------------------------------------------------
% ========= PLOT INTENSITY RATIO (figure invisible) ==========

%         hFig = figure('Visible','off','Color','w');
         ratio = abs(Extot_subs./Exinc_subs).^2;
%         plot(x*1e6, ratio,'k-','LineWidth',1.5);
%         xlabel('r [µm]'); ylabel('|E_{tot}/E_{inc}|^{2}');
%         grid on;
%         title(sprintf('gap = %.3f µm   |   L = %.2f µm', gap, L_ecran_subs));
% 
%         % ----------- SAUVEGARDE FIGURE -----------------------------
%         fNameImg = sprintf('gap_%0.4fum_L_%0.3fum.png', gap, L_ecran_subs);
%         exportgraphics(hFig, fullfile(outDir, fNameImg), 'Resolution', 300);
%         close(hFig);

        % ----------- SAUVEGARDE RAW DATA ---------------------------
        fNameMat = sprintf('gap_%0.4fum_L_%0.3fum.mat', gap, L_ecran_subs);
        save(fullfile(outDir, fNameMat), 'x', 'ratio', 'gap', 'L_ecran_subs');

        % ----------- MISE À JOUR MÉTADONNÉES ------------------------
       % fileNames(sampleIdx)     = fNameImg;
        gap_values_um(sampleIdx) = gap;
        L_values_um(sampleIdx)   = L_ecran_subs;
        sampleIdx = sampleIdx + 1;
end
save all_banque_new_04_07_25_NEW_full
end
% ---------------------- APRÈS LES BOUCLES ------------------------------
labelsTbl = table(fileNames, gap_values_um, L_values_um, ...
    'VariableNames',{'filename','gap_um','L_um'});
writetable(labelsTbl, fullfile(outDir,'labels.csv'));
save(fullfile(outDir,'labels.mat'), 'labelsTbl');

disp(['✅ Dataset prêt dans : ', outDir]);


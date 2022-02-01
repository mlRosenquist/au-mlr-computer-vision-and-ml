function [eigvec_now_2D]=irreg_QCUT_suppixel_15(image_now,suppix_num,m,model)


[LabelLine,suppixel,max_label,boundaries,LMean,AMean,BMean]=findSuppixParameters(image_now,m,suppix_num);

%Initialize Hamiltonian
H=zeros(max_label,max_label);

warning off
%% CALCULATE ALL NEIGHBOURS AND CHUNKS
[length_final_neighbours,length_first_neighbours,length_second_neighbours,length_third_neighbours,length_fourth_neighbours,length_fifth_neighbours,first_neighbourhood,second_neighbourhood,third_neighbourhood,fourth_neighbourhood,fifth_neighbourhood]=CalculateChunks(suppixel,max_label);

% CALCULATE DOT PRODUCT OF SUPERPIXELS' FEATURES
psi=[LMean'; AMean'; BMean'];
psi=psi/6540;


[ALL_MAPS2] = encode(psi,model);

%L2 normalization for features
norm=sqrt(sum(ALL_MAPS2.^2,2));
for ll=1:size(ALL_MAPS2,2)
    ALL_MAPS2(:,ll)=ALL_MAPS2(:,ll)./norm;
end
%Calculate the affinities as dot products
% ALL_DIST2=(ALL_MAPS2*ALL_MAPS2');

ALL_DIST2=1./(eps+squareform(pdist(ALL_MAPS2,'euclidean').^2));

% ALL_DIST2=1./(eps+squareform(pdist(psi','euclidean').^2));

% ASSIGN H DUE TO SYMMETRIC EQCUT MODEL
H2=assign_H(ALL_DIST2,ALL_DIST2*0,max_label,first_neighbourhood,second_neighbourhood,third_neighbourhood,fourth_neighbourhood,fifth_neighbourhood,length_first_neighbours,length_second_neighbours,length_third_neighbours,length_fourth_neighbours,length_fifth_neighbours,length_final_neighbours);
H2=-H2/min(min(H2));
%Assign Boundaries of Image as Background
for bnd_cnt=1:length(boundaries)
    H2(boundaries(bnd_cnt),boundaries(bnd_cnt))=H2(boundaries(bnd_cnt),boundaries(bnd_cnt))+1;
end

%Run QCUT
[eigvec2 eigval2]=eigs(sparse(H2),1,'SM');
eigvec142=eigvec2(:,1).^2;


eigvec_now_2D = sup2pixel( numel(image_now(:,:,1)), LabelLine,eigvec142 );
eigvec_now_2D = reshape( eigvec_now_2D, size(image_now,1), size(image_now,2) );

















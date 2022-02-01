%This function forms the Hamiltonian matrix used by QCUT, by the predefined
%neighbourhood information. The neighbourhood construction and
% relevant normalization is conducted as in EQCUT.

function H=assign_H(ALL_DIST,H,max_label,first_neighbourhood,second_neighbourhood,third_neighbourhood,fourth_neighbourhood,fifth_neighbourhood,length_first_neighbours,length_second_neighbours,length_third_neighbours,length_fourth_neighbours,length_fifth_neighbours,length_final_neighbours)

%For each superpixel normalize the affinities by symmetric EQCUT
%normalization for 5 neighbourhood groups
for label_counter=1:max_label
    neighbours=(first_neighbourhood{label_counter});
    length_neighbours=length_first_neighbours(label_counter);

    for neighbour_counter=1:length(neighbours)
        
        dist=ALL_DIST(label_counter,neighbours(neighbour_counter));
        mean_norm=(length_neighbours*length_final_neighbours(neighbours(neighbour_counter))).^2+(length_first_neighbours(neighbours(neighbour_counter))*length_final_neighbours(label_counter)).^2;
        UPDATE=dist/(mean_norm);
        H(label_counter,neighbours(neighbour_counter))=-UPDATE;
        H(label_counter,label_counter)=H(label_counter,label_counter)+UPDATE;
    end
    
    
    neighbours=(second_neighbourhood{label_counter});
    length_neighbours=length_second_neighbours(label_counter);
    
    for neighbour_counter=1:length(neighbours)
        dist=ALL_DIST(label_counter,neighbours(neighbour_counter));
        mean_norm=(length_neighbours*length_final_neighbours(neighbours(neighbour_counter))).^2+(length_second_neighbours(neighbours(neighbour_counter))*length_final_neighbours(label_counter)).^2;
        UPDATE=dist/(mean_norm);
        H(label_counter,neighbours(neighbour_counter))=-UPDATE;
        H(label_counter,label_counter)=H(label_counter,label_counter)+UPDATE;
    end
    
    
    neighbours=(third_neighbourhood{label_counter});
    length_neighbours=length_third_neighbours(label_counter);
    
    for neighbour_counter=1:length(neighbours)
        dist=ALL_DIST(label_counter,neighbours(neighbour_counter));
        mean_norm=(length_neighbours*length_final_neighbours(neighbours(neighbour_counter))).^2+(length_third_neighbours(neighbours(neighbour_counter))*length_final_neighbours(label_counter)).^2;
        UPDATE=dist/(mean_norm);
        H(label_counter,neighbours(neighbour_counter))=-UPDATE;
        H(label_counter,label_counter)=H(label_counter,label_counter)+UPDATE;
    end
    
    
    
    
    
    neighbours=(fourth_neighbourhood{label_counter});
    length_neighbours=length_fourth_neighbours(label_counter);
    
    for neighbour_counter=1:length(neighbours)
        dist=ALL_DIST(label_counter,neighbours(neighbour_counter));
        mean_norm=(length_neighbours*length_final_neighbours(neighbours(neighbour_counter))).^2+(length_fourth_neighbours(neighbours(neighbour_counter))*length_final_neighbours(label_counter)).^2;
        UPDATE=dist/(mean_norm);
        H(label_counter,neighbours(neighbour_counter))=-UPDATE;
        H(label_counter,label_counter)=H(label_counter,label_counter)+UPDATE;
    end
    
    
    
    neighbours=(fifth_neighbourhood{label_counter});
    length_neighbours=length_fifth_neighbours(label_counter);
    
    for neighbour_counter=1:length(neighbours)
        dist=ALL_DIST(label_counter,neighbours(neighbour_counter));
        mean_norm=(length_neighbours*length_final_neighbours(neighbours(neighbour_counter))).^2+(length_fifth_neighbours(neighbours(neighbour_counter))*length_final_neighbours(label_counter)).^2;
        UPDATE=dist/(mean_norm);
        H(label_counter,neighbours(neighbour_counter))=-UPDATE;
        H(label_counter,label_counter)=H(label_counter,label_counter)+UPDATE;
    end
    
    %
end
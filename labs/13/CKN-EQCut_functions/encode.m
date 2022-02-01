function [ALLMAPS] = encode(maps,model)
origmaps=maps;
for jj=1:model.nlayers
    for kk=1:jj-1
         maps=(encode_layer(origmaps,model.layer{kk}))'; %CAGLAR CHECK LATER
      end
        maps=encode_layer(maps,model.layer{jj});
   
end


ALLMAPS=maps;


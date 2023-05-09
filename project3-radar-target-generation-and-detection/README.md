# Radar Target Generation and Detection
## FP.1 Selection of Training, Guard cells and offset
``` matlab
%Select the number of Training Cells in both the dimensions.
Tr=8;
Td=8;

%Select the number of Guard Cells in both dimensions around the Cell under 
%test (CUT) for accurate estimation
Gr=4;
Gd=4;

% offset the threshold by SNR value in dB
off_set=1.4;
```

## FP.2 Implementation steps for the 2D CFAR process
``` matlab
% Loop over CUTs
for i = Tr+Gr+1:Nr-(Tr+Gr)
    for j = Td+Gd+1:Nd-(Td+Gd)
        % Loop over cells surrounding the CUT
        for p = i-(Tr+Gr) : i+(Tr+Gr)
            for q = j-(Td+Gd) : j+(Td+Gd)
                % Measure noise of training cells only
                if (abs(i-p) > Gr || abs(j-q) > Gd)
                    noise_level = noise_level + db2pow(RDM(p,q));
                end
            end
        end
        
        % Average 
        threshold = pow2db(noise_level/(2*Tr+2*Gr+1)*(2*Td+2*Gd+1) - (2*Gr+1)*(2*Gd+1));
        % Offset
        threshold = threshold + off_set;
        
        if (RDM(i,j) < threshold)
            RDM(i,j) = 0;
        else
            RDM(i,j) = 1;
        end
        
    end
end
```

## FP.3 Steps taken to suppress the non-thresholded cells at the edges
``` matlab
RDM(RDM!=1) = 0;
```
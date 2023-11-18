function [PSNR, SSIM] = evaluation(test_folder)
%     test_folder = 'results\LAL_BDP-llie-L150';
    gt_folder = 'C:\Datasets\LOLdataset\eval\eval15\high';
    test_list = dir(test_folder);

    avg_psnr = 0;
    avg_ssim = 0;
    avg_niqe = 0;

    count = 0;

    for i=3:length(test_list)
        count = count + 1;

        % load images
        f1 = fullfile(test_folder,  test_list(i).name);
        f2 = fullfile(gt_folder,  test_list(i).name);
        img_test = imread(f1);
        img_gt = imread(f2);

        % calculate metrics
        [peaksnr, snr] = psnr(img_test, img_gt);
        [ssimval,ssimmap] = ssim(img_test, img_gt);
        niqe_score = niqe(img_test);    
        
        
        avg_psnr=avg_psnr+peaksnr;
        avg_ssim=avg_ssim+ssimval;
        avg_niqe = avg_niqe + niqe_score;


        % print
%         fprintf('%d\n', count);
%         fprintf('PSNR=%.4f\tSSIM=%.4f\n', peaksnr, ssimval);
%         disp (f1);
%         disp (f2);
    end
    
    PSNR = avg_psnr/count;
    SSIM = avg_ssim/count;
    NIQE = avg_niqe/count;
	fprintf('Total\nPSNR=%.4f\tSSIM=%.4f\tNIQE=%.4f\n', PSNR, SSIM, NIQE);
    
end


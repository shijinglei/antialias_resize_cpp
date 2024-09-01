#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

using namespace std;

struct filter {
    float (*filter)(float x);
    float support;
};

static inline float sinc_filter(float x)
{
    if (x == 0.0)
        return 1.0;
    x = x * M_PI;
    return sin(x) / x;
}

static inline float antialias_filter(float x)
{
    /* lanczos (truncated sinc) */
    if (-3.0 <= x && x < 3.0)
        return sinc_filter(x) * sinc_filter(x/3);
    return 0.0;
}

static struct filter ANTIALIAS = { antialias_filter, 3.0 };


void LanczosResize(cv::Mat &imOut, const cv::Mat &imIn) {
    struct filter *filterp;
    float support_x, support_y, scale_x, scale_y, filterscale_x, filterscale_y;
    float center_x, center_y, ww_x, ww_y, ss_x, ss_y, ss, ymin, ymax, xmin, xmax;
    int xx, yy, x, y, b;
    float *k_x, *k_y;

    /* check filter */
    filterp = &ANTIALIAS;

    filterscale_x = scale_x = (float) imIn.cols / imOut.cols;
    filterscale_y = scale_y = (float) imIn.rows / imOut.rows;

    /* determine support size (length of resampling filter) */
    support_x = support_y = filterp->support;

    if (filterscale_x < 1.0) {
        filterscale_x = 1.0;
        support_x = 0.5;
    }
    support_x = support_x * filterscale_x;

    if (filterscale_y < 1.0) {
        filterscale_y = 1.0;
        support_y = 0.5;
    }
    support_y = support_y * filterscale_y;

    /* coefficient buffer (with rounding safety margin) */
    k_x = (float*)malloc(((int) support_x * 2 + 2) * sizeof(float));
    if (!k_x) {
        cout << "error in malloc." << endl;
        return;
    }
    k_y = (float*)malloc(((int) support_y * 2 + 2) * sizeof(float));
    if (!k_y) {
        cout << "error in malloc." << endl;
        return;
    }

    for (yy = 0; yy < imOut.rows; yy++) {
        for(xx = 0; xx < imOut.cols; xx++) {
            center_y = (yy + 0.5) * scale_y;
            ww_y = 0.0;
            ss_y = 1.0 / filterscale_y;
            center_x = (xx + 0.5) * scale_x;
            ww_x = 0.0;
            ss_x = 1.0 / filterscale_x;
            
            /* calculate filter_y weights */
            ymin = floor(center_y - support_y);
            if (ymin < 0.0)
                ymin = 0.0;
            ymax = ceil(center_y + support_y);
            if (ymax > (float) imIn.rows)
                ymax = (float) imIn.rows;
            for (y = (int) ymin; y < (int) ymax; y++) {
                float w = filterp->filter((y - center_y + 0.5) * ss_y) * ss_y;
                k_y[y - (int) ymin] = w;
                ww_y = ww_y + w;
            }
            if (ww_y == 0.0)
                ww_y = 1.0;
            else
                ww_y = 1.0 / ww_y;
            
            /* calculate filter_x weights */
            xmin = floor(center_x - support_x);
            if (xmin < 0.0)
                xmin = 0.0;
            xmax = ceil(center_x + support_x);
            if (xmax > (float) imIn.cols)
                xmax = (float) imIn.cols;
            for (x = (int) xmin; x < (int) xmax; x++) {
                float w = filterp->filter((x - center_x + 0.5) * ss_x) * ss_x;
                k_x[x - (int) xmin] = w;
                ww_x = ww_x + w;
            }
            if (ww_x == 0.0)
                ww_x = 1.0;
            else
                ww_x = 1.0 / ww_x;
            
            for (int c = 0; c < imOut.channels(); c++) {
                ss = 0.0;
                for (y = (int) ymin; y < (int) ymax; y++) {
                    for (x = (int) xmin; x < (int) xmax; x++) {
                        float *src_pix = (float*)imIn.data + (y * imIn.cols + x) * imIn.channels() + c;
                        ss = ss + (*src_pix ) * k_y[y - (int) ymin] * k_x[x - (int) xmin];
                    }
                }
                float *dst_pix = (float*)imOut.data + (yy * imOut.cols + xx) * imOut.channels() + c;
                *dst_pix = ss * ww_x * ww_y;
            }
        }
    }
        
    free(k_x);
    free(k_y);
}

/**
 * @file rtl-sdr-fft.cpp
 *
 * RTL SDR FFT implementation heavily based on rtl_power.c from rtlsdr lib.
 * It uses a little bit of C++ plus complex calculus.
 *
 * @author Lukasz Wiecaszek <lukasz.wiecaszek@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 */

/*===========================================================================*\
 * system header files
\*===========================================================================*/
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <signal.h>
#include <math.h>

#include <vector>
#include <chrono>
#include <thread>

#include <rtl-sdr.h>

/*===========================================================================*\
 * project header files
\*===========================================================================*/
#include "strtointeger.hpp"
#include "power_of_two.hpp"
#include "fixq15.hpp"
#include "complex.hpp"
#include "fft.hpp"
#include "pipeline.hpp"
#include "ringbuffer.hpp"

/*===========================================================================*\
 * preprocessor #define constants and macros
\*===========================================================================*/
#define FFT_SIZE_MAX    (8 * 1024)
#define IQBUF_SIZE      (FFT_SIZE_MAX * 2)
#define IDLE_LOOPS_NUM  (1)

/*===========================================================================*\
 * local type definitions
\*===========================================================================*/
template<typename T>
struct buffer : public ymn::pipeline::buffer
{
    explicit buffer() :
        ymn::pipeline::buffer{},
        vector()
    {
    }

    explicit buffer(std::size_t size) :
        ymn::pipeline::buffer{},
        vector(size)
    {
    }

    std::vector<T> vector;
};

using iq_t = ymn::complex<ymn::fixq15>;
using iq_buffer_uptr = std::unique_ptr<buffer<iq_t>>;

iq_buffer_uptr to_iq_buffer_uptr(ymn::pipeline::buffer_uptr&& p)
{
    return iq_buffer_uptr{static_cast<buffer<iq_t>*>(p.release())};
}

/*===========================================================================*\
 * global object definitions
\*===========================================================================*/

/*===========================================================================*\
 * local function declarations
\*===========================================================================*/
static void print_usage(const char* progname);
static void signal_handler(int signum);
static void install_signal_handler(void);
static void remove_dc(iq_t* iqbuf, const std::size_t N);
static void print_fft(FILE *fp, uint32_t fc, uint32_t bw, iq_t* iqbuf, const std::size_t N);
static int verbose_device_search(const char *s);

/*===========================================================================*\
 * local object definitions
\*===========================================================================*/
static rtlsdr_dev_t *rtlsdr_device = NULL;
static uint8_t iqbuf_u8[IQBUF_SIZE];
static std::unique_ptr<iq_t[]> e_2pi_i;
static std::unique_ptr<ymn::pipeline> pipeline;

/*===========================================================================*\
 * inline function definitions
\*===========================================================================*/
static inline iq_buffer_uptr get_iq_buffer_uptr(ymn::iringbuffer<ymn::pipeline::buffer_uptr>* irb)
{
    ymn::pipeline::buffer_uptr buf_uptr;

    long read_status = irb->read(std::move(buf_uptr));
    if (read_status != 1)
        return nullptr;

    return to_iq_buffer_uptr(std::move(buf_uptr));
}

inline void generate_e_2pi_i(iq_t* e, const std::size_t N)
{
    for (std::size_t i = 0; i < N; ++i) {
        double x = 2.0 * M_PI * i / N;
        e[i].real(static_cast<ymn::fixq15>(round(Q15 * cos(x))));
        e[i].imag(static_cast<ymn::fixq15>(round(Q15 * sin(x))));
    }
}

template<std::size_t N>
inline void generate_e_2pi_i(iq_t (&e)[N])
{
    static_assert(ymn::is_power_of_two(N), "N must be power of 2");
    generate_e_2pi_i(e, N);
}

/*===========================================================================*\
 * public function definitions
\*===========================================================================*/
int main(int argc, char *argv[])
{
    int status;
    uint32_t frequency = 0;
    uint32_t bandwidth = 2000000;
    int fft_size = 2048;
    FILE* fp;
    int dev_index;

    install_signal_handler();

    static const struct option long_options[] = {
        {"frequency", required_argument, 0, 'f'},
        {"bandwidth", required_argument, 0, 'b'},
        {"fft-size",  required_argument, 0, 'n'},
        {0, 0, 0, 0}
    };

    for (;;) {
        int c = getopt_long(argc, argv, "f:b:n:", long_options, 0);
        if (c == -1)
            break;

        switch (c) {
            case 'f':
                if (ymn::strtointeger(optarg, frequency) != ymn::strtointeger_conversion_status_e::success) {
                    fprintf(stderr, "Cannot convert '%s' to integer\n", optarg);
                    exit(EXIT_FAILURE);
                }
                break;

            case 'b':
                if (ymn::strtointeger(optarg, bandwidth) != ymn::strtointeger_conversion_status_e::success) {
                    fprintf(stderr, "Cannot convert '%s' to integer\n", optarg);
                    exit(EXIT_FAILURE);
                }
                break;

            case 'n':
                if (ymn::strtointeger(optarg, fft_size) != ymn::strtointeger_conversion_status_e::success) {
                    fprintf(stderr, "Cannot convert '%s' to integer\n", optarg);
                    exit(EXIT_FAILURE);
                }
                break;

            default:
                /* do nothing */
                break;
        }
    }

    if (argc > optind) {
        fp = fopen(argv[optind], "w");
        if (fp == NULL) {
            fprintf(stderr, "Cannot create '%s'\n", argv[optind]);
            exit(EXIT_FAILURE);
        }
    }
    else
        fp = stdout;

    if (frequency == 0) {
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    if (fft_size & (fft_size - 1)) {
        fprintf(stderr, "fft_size (%u) must be power of 2\n", fft_size);
        exit(EXIT_FAILURE);
    }

    if (fft_size > FFT_SIZE_MAX) {
        fprintf(stderr, "fft_size (%u) is too big (max supported is set to %u)\n",
            fft_size, FFT_SIZE_MAX);
        exit(EXIT_FAILURE);
    }

    dev_index = verbose_device_search("0");
    if (dev_index < 0)
        exit(EXIT_FAILURE);

    fprintf(stderr, "Opening device #%d\n", dev_index);
    status = rtlsdr_open(&rtlsdr_device, (uint32_t)dev_index);
    if (status < 0) {
        fprintf(stderr, "Failed to open rtlsdr device #%d\n", dev_index);
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, " - done\n");

    fprintf(stderr, "Setting tuner gain to automatic\n");
    status = rtlsdr_set_tuner_gain_mode(rtlsdr_device, 0);
    if (status) {
        fprintf(stderr, "rtlsdr_set_tuner_gain_mode(0) failed\n");
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, " - done\n");

    /* Reset endpoint before we start reading from it (mandatory) */
    fprintf(stderr, "Resseting rtlsdr buffers\n");
    status = rtlsdr_reset_buffer(rtlsdr_device);
    if (status) {
        fprintf(stderr, "rtlsdr_reset_buffer() failed\n");
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, " - done\n");

    fprintf(stderr, "Setting center frequency to %u Hz\n", frequency);
    status = rtlsdr_set_center_freq(rtlsdr_device, frequency);
    if (status) {
        fprintf(stderr, "rtlsdr_set_center_freq(%u) failed\n", frequency);
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, " - done\n");

    fprintf(stderr, "Setting sample rate to %u Hz\n", bandwidth);
    status = rtlsdr_set_sample_rate(rtlsdr_device, bandwidth);
    if (status) {
        fprintf(stderr, "rtlsdr_set_sample_rate(%u) failed\n", bandwidth);
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, " - done\n");

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    e_2pi_i = std::make_unique<iq_t[]>(fft_size);
    generate_e_2pi_i(e_2pi_i.get(), fft_size);

    auto producer = [&](ymn::iringbuffer<ymn::pipeline::buffer_uptr>* irb, ymn::oringbuffer<ymn::pipeline::buffer_uptr>* orb){

        assert(irb == nullptr);
        assert(orb != nullptr);

        int status;
        int n_read;
        static std::size_t counter = 0;

        status = rtlsdr_read_sync(rtlsdr_device, iqbuf_u8, sizeof(iqbuf_u8), &n_read);
        if (status) {
            fprintf(stderr, "rtlsdr_read_sync(%zu) failed\n", sizeof(iqbuf_u8));
            return false;
        }

        if (n_read != sizeof(iqbuf_u8)) {
            fprintf(stderr, "rtlsdr_read_sync(%zu) dropped samples - received %d\n",
                sizeof(iqbuf_u8), n_read);
            return true;
        }

        if (counter++ < IDLE_LOOPS_NUM)
            return true;

        for (std::size_t offset = 0; offset < sizeof(iqbuf_u8); offset += (fft_size * 2)) {
            iq_buffer_uptr iqbuf_uptr = std::make_unique<buffer<iq_t>>(fft_size);
            iq_t* iqbuf = iqbuf_uptr->vector.data();
            const uint8_t* src = &iqbuf_u8[offset];

            /* scale [0, 255] -> [-127, 128] */
            /* scale [-127, 128] -> [-32512, 32768] */
            for (int i = 0; i < fft_size; ++i) {
                iqbuf[i].real((src[2 * i + 0] - 127) * 256);
                iqbuf[i].imag((src[2 * i + 1] - 127) * 256);
				fprintf(fp, "%d   %d\n", src[2 * i + 0] - 127, src[2 * i + 1] - 127);
            }

            long write_status = orb->write(std::move(iqbuf_uptr));
            if (write_status != 1) {
               fprintf(stderr, "%s: orb->write() failed\n", __PRETTY_FUNCTION__);
               fprintf(stderr, "%s\n", orb->to_string().c_str());
            }
        }

        return true;
    };

    auto fft_stage = [&](ymn::iringbuffer<ymn::pipeline::buffer_uptr>* irb, ymn::oringbuffer<ymn::pipeline::buffer_uptr>* orb){

        assert(irb != nullptr);
        assert(orb == nullptr);

        iq_buffer_uptr iqbuf_uptr = get_iq_buffer_uptr(irb);
        if (!iqbuf_uptr)
            return false;

        remove_dc(iqbuf_uptr->vector.data(), fft_size); // Is it necessary?
        fft(iqbuf_uptr->vector.data(), e_2pi_i.get(), fft_size);

        //fprintf(fp, "%s\n", irb->to_string().c_str());

        print_fft(fp, frequency, bandwidth, iqbuf_uptr->vector.data(), fft_size);

        return true;
    };

    ymn::pipeline::stage_function functions[] = {producer, fft_stage};
    pipeline = std::make_unique<ymn::pipeline>(functions, 42);

    pipeline->start();
    pipeline->join();

    rtlsdr_close(rtlsdr_device);

    if (fp != stdout)
        fclose(fp);

    return 0;
}

/*===========================================================================*\
 * local function definitions
\*===========================================================================*/
static void print_usage(const char* progname)
{
    fprintf(stdout, "usage: %s -f <frequency> [-b <bandwidth>] [-n <fft_size>] [<filename>]\n", progname);
    fprintf(stdout, " options:\n");
    fprintf(stdout, "  -f <frequency>  --frequency=<frequency> : center frequency to tune to\n");
    fprintf(stdout, "  -b <bandwidth>  --bandwidth=<bandwidth> : bandwidth to be scanned (default: 2 MHz)\n");
    fprintf(stdout, "  -n <fft size>   --fft-size=<fft size>   : fft size (default: 2048)\n");
    fprintf(stdout, "  <filename>                              : print output values to this file (default: stdout)\n");
}

static void signal_handler(int signum)
{
    fprintf(stderr, "caught signal %d, terminating ...\n", signum);
    if (pipeline != nullptr)
        pipeline->stop();
    fprintf(stderr, "done\n");
}

static void install_signal_handler(void)
{
    struct sigaction sigact;

    sigact.sa_handler = signal_handler;
    sigemptyset(&sigact.sa_mask);
    sigact.sa_flags = 0;

    sigaction(SIGINT, &sigact, NULL);
    sigaction(SIGTERM, &sigact, NULL);
    sigaction(SIGQUIT, &sigact, NULL);
    sigaction(SIGPIPE, &sigact, NULL);
}

static void remove_dc(iq_t* iqbuf, const std::size_t N)
{
    iq_t sum(0, 0);

    for (std::size_t n = 0; n < N; ++n)
        sum += iqbuf[n];

    iq_t average = sum / ymn::fixq15(N * Q15);

    //fprintf(stderr, "dc component: %s\n", average.to_string().c_str());

    if (average == iq_t(0, 0))
        return;

    for (std::size_t n = 0; n < N; ++n)
        iqbuf[n] -= average;
}

static void print_fft(FILE *fp, uint32_t fc, uint32_t bw, iq_t* iqbuf, const std::size_t N)
{
	#if 0
    uint32_t f = fc - (bw / 2);
    uint32_t f_step = bw / N;

    for (std::size_t n = 0; n < N; ++n, f += f_step)
        fprintf(fp, "%8zu\t\t%8u Hz\t\t%8ld\t\t%8ld\t\t%8ld\n",
            n,
            f,
            iqbuf[n].real().value() / Q15,
            iqbuf[n].imag().value() / Q15,
            iqbuf[n].norm().value() / Q15);
			#endif
}

static int verbose_device_search(const char *s)
{
    int i, device_count, device, offset;
    char *s2;
    char vendor[256], product[256], serial[256];

    device_count = rtlsdr_get_device_count();
    if (!device_count) {
        fprintf(stderr, "No supported devices found\n");
        return -1;
    }

    fprintf(stderr, "Found %d device(s):\n", device_count);
    for (i = 0; i < device_count; i++) {
        rtlsdr_get_device_usb_strings(i, vendor, product, serial);
        fprintf(stderr, "  %d:  %s, %s, SN: %s\n", i, vendor, product, serial);
    }
    fprintf(stderr, "\n");

    /* does string look like raw id number */
    device = (int)strtol(s, &s2, 0);
    if (s2[0] == '\0' && device >= 0 && device < device_count) {
        fprintf(stderr, "Using device %d: %s\n",
            device, rtlsdr_get_device_name((uint32_t)device));
        return device;
    }

    /* does string exact match a serial */
    for (i = 0; i < device_count; i++) {
        rtlsdr_get_device_usb_strings(i, vendor, product, serial);
        if (strcmp(s, serial) != 0) {
            continue;}
        device = i;
        fprintf(stderr, "Using device %d: %s\n",
            device, rtlsdr_get_device_name((uint32_t)device));
        return device;
    }

    /* does string prefix match a serial */
    for (i = 0; i < device_count; i++) {
        rtlsdr_get_device_usb_strings(i, vendor, product, serial);
        if (strncmp(s, serial, strlen(s)) != 0) {
            continue;}
        device = i;
        fprintf(stderr, "Using device %d: %s\n",
            device, rtlsdr_get_device_name((uint32_t)device));
        return device;
    }

    /* does string suffix match a serial */
    for (i = 0; i < device_count; i++) {
        rtlsdr_get_device_usb_strings(i, vendor, product, serial);
        offset = strlen(serial) - strlen(s);
        if (offset < 0) {
            continue;}
        if (strncmp(s, serial+offset, strlen(s)) != 0) {
            continue;}
        device = i;
        fprintf(stderr, "Using device %d: %s\n",
            device, rtlsdr_get_device_name((uint32_t)device));
        return device;
    }

    fprintf(stderr, "No matching devices found\n");

    return -1;
}


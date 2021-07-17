import numpy as np
import matplotlib.pyplot as plt


def task6_4():
  # 定義
  fs = 44100
  fft_size = 65536
  fft_size2 = 65536 * 16
  fc = 100
  w = np.arange(fft_size)*fs / fft_size
  w2 = np.arange(fft_size2)*fs / fft_size2

  fc_index = round(fft_size*fc/fs) + 1
  amp_spec = np.ones(fft_size//2+1)
  amp_spec[fc_index:] = 0

  spec = np.block([amp_spec, amp_spec[-2:0:-1]])
  impulse_response = np.fft.fftshift(np.real(np.fft.ifft(spec)))

  # plot
  plt.plot(
      w,
      np.abs(np.fft.fft(impulse_response, fft_size)),
      marker=".",
      ls="None",
  )
  plt.plot(w2, np.abs(np.fft.fft(impulse_response, fft_size2)))
  plt.xlim([95, 105])
  plt.xlabel("Freq.")
  plt.ylabel("Amp.")
  plt.show()


if __name__ == "__main__":
    task6_4()

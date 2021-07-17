import numpy as np
import matplotlib.pyplot as plt
from func import MyBlackman


def task6_5():
  # 定義
  fs = 44100
  fft_size = 65536
  fft_size2 = 65536 * 16
  fc = 100
  w2 = np.arange(fft_size2)*fs / fft_size2

  fc_index = round(fft_size*fc/fs) + 1
  amp_spec = np.ones(fft_size//2+1)
  amp_spec[fc_index:] = 0

  spec = np.block([amp_spec, amp_spec[-2:0:-1]])
  impulse_response = np.fft.fftshift(np.real(np.fft.ifft(spec)))

  half_N = 32767
  window_index = np.arange(fft_size//2 - half_N, fft_size//2 + half_N)
  h = impulse_response[window_index] * MyBlackman(half_N*2)


  # plot
  plt.plot(
    w2,
    np.abs(np.fft.fft(impulse_response, fft_size2)),
    label="use impulse_response",
    ls="--",
  )
  plt.plot(
    w2,
    np.abs(np.fft.fft(h, fft_size2)),
    label="use Blackman win")
  plt.xlim([90, 110])
  plt.xlabel("Freq.")
  plt.ylabel("Amp.")
  plt.show()


if __name__ == "__main__":
    task6_5()

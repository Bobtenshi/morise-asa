import numpy as np
import matplotlib.pyplot as plt
from func import MyBlackman,MinimumPhase

def task6_6():
  # 定義
  fs = 44100
  fft_size = 65536
  fft_size2 = 65536 * 16
  fc = 100
  w = np.arange(fft_size)*fs / fft_size
  w2 = np.arange(fft_size2)*fs / fft_size2

  fc_index = round(fft_size*fc/fs) + 1
  amp_spec = np.ones(fft_size//2+1)
  amp_spec[fc_index + 1:] = 0.01

  spec = np.block([amp_spec, amp_spec[-2:0:-1]])
  impulse_response = np.fft.fftshift(np.real(np.fft.ifft(spec)))
  impulse_response[0] = 0
  impulse_response[1:fft_size] = impulse_response[1:fft_size] * MyBlackman(fft_size - 1)

  minimum_phase_response = MinimumPhase(impulse_response)

  # plot
  plt.plot(
    w2,
    np.abs(np.fft.fft(impulse_response, fft_size2)),
    label="use impulse_response",
    ls="--",
  )
  plt.plot(
    w2,
    np.abs(np.fft.fft(minimum_phase_response, fft_size2)),
    label="use minimum_phase_response")
  plt.xlim([90, 110])
  plt.xlabel("Freq.")
  plt.ylabel("Amp.")
  plt.legend()
  plt.show()


if __name__ == "__main__":
    task6_6()

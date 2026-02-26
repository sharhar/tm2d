import numpy as np
import matplotlib.pyplot as plt

example_ctf = np.load("data/results/example_ctf.npy")
example_template = np.load("data/results/example_template.npy")
example_template2 = np.load("data/results/example_template2.npy")
example_comparison = np.load("data/results/example_comparison.npy")
example_comparison2 = np.load("data/results/example_comparison2.npy")
template = np.load("data/results/template_0.npy")
comparison = np.load("data/results/comparison_0.npy")
mip = np.load("data/results/mip_0.npy")
z_score = np.load("data/results/z_score_0.npy")

example_ctf_ref = np.load("data/results_REF3/example_ctf.npy")
example_template_ref = np.load("data/results_REF3/example_template.npy")
example_template2_ref = np.load("data/results_REF3/example_template2.npy")
example_comparison_ref = np.load("data/results_REF3/example_comparison.npy")
example_comparison2_ref = np.load("data/results_REF3/example_comparison2.npy")
template_ref = np.load("data/results_REF3/template_0.npy")
comparison_ref = np.load("data/results_REF3/comparison_0.npy")
mip_ref = np.load("data/results_REF3/mip_0.npy")
z_score_ref = np.load("data/results_REF3/z_score_0.npy")

#example_ctf_diff = example_ctf - example_ctf_ref
example_template_diff = example_template - example_template_ref
example_template2_diff = example_template2 - example_template2_ref
example_comparison_diff = example_comparison - example_comparison_ref
example_comparison2_diff = example_comparison2 - example_comparison2_ref
comparison_diff = comparison - comparison_ref
mip_diff = mip - mip_ref
z_score_diff = z_score - z_score_ref

np.save("comparison.npy", example_comparison)
np.save("comparison_ref.npy", example_comparison_ref)
np.save("comparison_diff.npy", example_comparison_diff)

np.save("comparison_fft.npy", np.fft.fftshift(np.fft.fft2(example_comparison)))
np.save("comparison_ref_fft.npy", np.fft.fftshift(np.fft.fft2(example_comparison_ref)))
np.save("comparison_diff_fft.npy", np.fft.fftshift(np.fft.fft2(example_comparison)) - np.fft.fftshift(np.fft.fft2(example_comparison_ref)))


np.save("comparison2.npy", example_comparison2)
np.save("comparison2_ref.npy", example_comparison2_ref)
np.save("comparison2_diff.npy", example_comparison2_diff)

np.save("comparison2_fft.npy", np.fft.fftshift(np.fft.fft2(example_comparison2)))
np.save("comparison2_ref_fft.npy", np.fft.fftshift(np.fft.fft2(example_comparison2_ref)))
np.save("comparison2_diff_fft.npy", np.fft.fftshift(np.fft.fft2(example_comparison2)) - np.fft.fftshift(np.fft.fft2(example_comparison_ref)))

np.save("template.npy", example_template)
np.save("template_ref.npy", example_template_ref)
np.save("template_diff.npy", example_template - example_template_ref)

np.save("template_fft.npy", np.fft.fftshift(np.fft.fft2(template)))
np.save("template_ref_fft.npy", np.fft.fftshift(np.fft.fft2(template_ref)))
np.save("template_diff_fft.npy", np.fft.fftshift(np.fft.fft2(template - template_ref)))

exit()

plt.clf()
plt.imshow(example_ctf_diff)
plt.colorbar()
plt.title("Difference in CTF")
plt.savefig("data/results/diff_example_ctf.png")

plt.clf()
plt.imshow(example_template_diff)
plt.colorbar()
plt.title("Difference in template")
plt.savefig("data/results/diff_example_template.png")

plt.clf()
plt.imshow(comparison_diff)
plt.colorbar()
plt.title("Difference in comparison")
plt.savefig("data/results/diff_comparison.png")

plt.clf()
plt.imshow(mip_diff)
plt.colorbar()
plt.title("Difference in MIP")
plt.savefig("data/results/diff_mip.png")

plt.clf()
plt.imshow(z_score_diff)
plt.colorbar()
plt.title("Difference in z-score")
plt.savefig("data/results/diff_z_score.png")
import os
import sol4


def main():

    pano_vid = 'pano_vid.mp4'
    exp_no_ext = pano_vid.split('.')[0]
    os.system('mkdir dump')
    os.system('mkdir dump/%s' % exp_no_ext)
    os.system('ffmpeg -i videos/%s dump/%s/%s%%03d.jpg' % (pano_vid, exp_no_ext, exp_no_ext))

    panorama_generator = sol4.PanoramicVideoGenerator('dump/%s/' % exp_no_ext, exp_no_ext, 2100)
    panorama_generator.align_images(translation_only=False)
    panorama_generator.generate_panoramic_images(16)

    panorama_generator.save_panoramas_to_video()


if __name__ == '__main__':
    main()

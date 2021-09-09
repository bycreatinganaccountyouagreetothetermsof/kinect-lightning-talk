#!/usr/bin/env python3

import cv2
import freenect as kinect
import numpy as np
import pygame as pg

def info(log,a):
    print(log,a,a.dtype,a.size,a.shape,a.max(),a.min(),flush=True)

pg.init()
pg.mouse.set_visible(False)
#screen = pg.display.set_mode(resolution, display_mode)
screen = pg.display.set_mode((640,480),pg.FULLSCREEN)
resolution = screen.get_size()
clock = pg.time.Clock()

is_running = True

ctl_near = 400
ctl_far = 2000
ctl_invert = False
ctl_clip = False
ctl_march = False
ctl_gray = True
ctl_detail = False
ctl_chromakey = False

video_format = kinect.VIDEO_RGB
depth_format = kinect.DEPTH_MM

def solid_color(r,g,b):
    x, y = (640,480) #resolution
    solid = np.full((y,x,3),(r,g,b),dtype=np.uint8)
    return solid

solid_frame = solid_color(0,255,0)

print("first frames",flush=True)
d, td = kinect.sync_get_depth(format=depth_format)
print("depth",flush=True)
v, tv = kinect.sync_get_video() #format=video_format)
print("video",flush=True)

print("enter main loop",flush=True)

while is_running:
    frame = None
    for event in pg.event.get():
        if event.type == pg.QUIT:
            is_running = False
        if event.type == pg.KEYDOWN:
            print("keypress", event.key, flush=True)
            if event.key == pg.K_ESCAPE:
                is_running = False
            if event.key == pg.K_k:
                ctl_chromakey = not ctl_chromakey
            if event.key == pg.K_i:
                video_format = kinect.VIDEO_IR_10BIT
                depth_format = kinect.DEPTH_MM
            if event.key == pg.K_r:
                ctl_march = False
                video_format = kinect.VIDEO_RGB
                depth_format = kinect.DEPTH_REGISTERED # also in mm
            if event.key == pg.K_d:
                video_format = None
                depth_format = kinect.DEPTH_MM
            if event.key == pg.K_f:
                ctl_detail = not ctl_detail
            if event.key == pg.K_m:
                ctl_march = not ctl_march
            if event.key == pg.K_c:
                ctl_clip = not ctl_clip
            if event.key == pg.K_v:
                ctl_invert = not ctl_invert
            if event.key == pg.K_g:
                ctl_gray = not ctl_gray
            if event.key == pg.K_UP:
                ctl_far += 100
            if event.key == pg.K_DOWN:
                ctl_far -= 100
            if event.key == pg.K_LEFT:
                ctl_near -= 100
            if event.key == pg.K_RIGHT:
                ctl_near += 100

    if ctl_clip:
        d, td = kinect.sync_get_depth(format=depth_format)

    if ctl_chromakey: # for input into video mixer
        v, tv = solid_frame.copy(), td
        ctl_clip = True
        ctl_march = False
        ctl_detail = False
        ctl_gray = False
    elif video_format is not None:
        v, tv = kinect.sync_get_video(format=video_format)
    else: # video_format is None, use depth data
        if not ctl_clip:
            d, td = kinect.sync_get_depth(format=depth_format)
        #v, tv = (np.log2(d.copy())*161).astype(np.uint16), td
        v, tv = (np.log2(d)*161).astype(np.uint16), td
        np.clip(v,0,2047,v)

    if ctl_march:
        second = 1000*2048
        cycle = int(td/(1.5*second))
        v += cycle

    if ctl_gray and video_format is not kinect.VIDEO_RGB:
        if video_format is kinect.VIDEO_IR_10BIT:
            v = (v >> 2).astype(np.uint8)
        else:
            v = (v >> 3).astype(np.uint8)
        v = cv2.cvtColor(v,cv2.COLOR_GRAY2RGB)

    if ctl_clip:
        v[d < ctl_near] = 0
        v[d > ctl_far] = 0
        #d[d < ctl_near] = 0
        #d[d > ctl_far] = 0
        #if ctl_gray and ctl_detail:
        #    mid = (ctl_far-ctl_near)/2
        #    d = np.abs(d.astype(np.int16)-ctl_near-mid)
        #    d /= d.max()
        #    if v.shape > d.shape:
        #        d = np.reshape(1.0-d,(480,640,1))
        #    v = (v*d).astype(np.uint8)
        #else:
        #    if not ctl_invert:
        #        v[d == 0] = 0
        #    elif ctl_invert:
        #        v[d != 0] = 0

    frame = cv2.resize(v,resolution)
    frame = cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame = pg.pixelcopy.make_surface(frame)
    screen.blit(frame, (0, 0))
    pg.display.update()

print("normal exit",flush=True)

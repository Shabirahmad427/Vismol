#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  vismol_gtkwidget.py
#  
#  Copyright 2022 Carlos Eduardo Sequeiros Borja <casebor@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

from vismol.utils.debug import dprint
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk
from gi.repository import GdkPixbuf

from logging import getLogger
from vismol.libgl.vismol_glcore import VismolGLCore
from vismol.gui.filechooser import FileChooser
logger = getLogger(__name__)

from OpenGL.GL import *
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageChops
import os
import json
import sys

# [EN] macOS/Quartz fix: GTK3's Quartz backend has a long-standing bug
# where realizing a Gtk.GLArea corrupts Cairo/window compositing for the
# ENTIRE containing NSWindow, leaving it permanently blank (reproduces
# with a bare Gtk.Window containing nothing but a GLArea -- no
# EasyHybrid/Vismol code involved). GTK4 has a real fix (GSK_RENDERER=gl)
# but GTK3 has no equivalent, and running under XQuartz/X11 instead of
# Quartz was also tried and rejected (GTK3's X11 GL path segfaults on
# first draw on macOS). Neither backend has a working GtkGLArea on this
# platform. Worked around by rendering OpenGL OUTSIDE of GTK's own
# compositing entirely on macOS: VismolGTKWidget becomes a plain
# Gtk.DrawingArea there, owning a hidden GLFW window purely to hold an
# OpenGL context that's never attached to any GTK/AppKit view (see
# _OffscreenGLContext below); the finished frame is read back with
# glReadPixels and blitted into the DrawingArea's Cairo context. Linux/
# Windows are completely unaffected -- this constant is decided once, at
# import time, and every macOS-only code path below is gated behind it.
_IS_MACOS = sys.platform == "darwin"
_WidgetBase = Gtk.GLArea

if _IS_MACOS:
    import glfw
    import cairo
    _WidgetBase = Gtk.DrawingArea

    class _OffscreenGLContext:
        """ Owns a hidden GLFW window purely to hold an OpenGL 3.3 core
            context and its default framebuffer -- never shown, never
            attached to any GTK/AppKit view, so it never touches GTK's
            Quartz compositing. Resized to match the DrawingArea's
            current allocation before every draw (see
            VismolGTKWidget._draw_macos).
        """

        def __init__(self, width, height):
            if not glfw.init():
                raise RuntimeError("Failed to initialize GLFW for the offscreen macOS GL context.")
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
            glfw.window_hint(glfw.DEPTH_BITS, 24)
            glfw.window_hint(glfw.ALPHA_BITS, 8)
            self._window = glfw.create_window(max(int(width), 1), max(int(height), 1),
                                               "", None, None)
            if not self._window:
                glfw.terminate()
                raise RuntimeError("Failed to create the hidden GLFW window for the offscreen macOS GL context.")

        def make_current(self):
            glfw.make_context_current(self._window)

        def resize(self, width, height):
            glfw.set_window_size(self._window, max(int(width), 1), max(int(height), 1))


def _patch_gl_line_width():
    """ Apple's GL only guarantees glLineWidth(1.0) under a core profile
        (GL_ALIASED_LINE_WIDTH_RANGE is commonly just [1.0, 1.0]). The
        codebase calls glLineWidth() with wider values (2-5) in ~34
        places across glaxis.py, representations.py and selection_box.py
        for the axis gizmo, selection box and wire representations.
        Rather than edit every call site, patched centrally here: every
        one of those modules does "from OpenGL import GL" and calls
        GL.glLineWidth(...), i.e. they all look up the function by
        attribute on the *same shared* OpenGL.GL module object at call
        time rather than binding their own copy at import time -- so
        patching that one attribute once covers every call site. Not
        gated to macOS: it's purely defensive (clamps the requested
        width to whatever the driver actually reports), a no-op on
        Linux/Mesa/NVIDIA where the supported range is normally wide.
    """
    from OpenGL import GL

    if getattr(GL.glLineWidth, "_vismol_patched", False):
        return

    original_glLineWidth = GL.glLineWidth

    def _clamped_glLineWidth(width):
        try:
            lo, hi = GL.glGetFloatv(GL.GL_ALIASED_LINE_WIDTH_RANGE)
            width = min(max(width, lo), hi)
        except Exception:
            pass
        return original_glLineWidth(width)

    _clamped_glLineWidth._vismol_patched = True
    GL.glLineWidth = _clamped_glLineWidth


class VismolGTKWidget(_WidgetBase):
    """ Object that contains the GLArea from GTK3+ (Gtk.GLArea on Linux/
        Windows; a plain Gtk.DrawingArea + offscreen GLFW context on
        macOS, see _IS_MACOS above). It needs a vertex and shader to be
        created, maybe later I"ll add a function to change the shaders.
    """

    def __init__(self, vismol_session=None, width=640.0, height=420.0):
        """ Class initialiser
        """
        super(VismolGTKWidget, self).__init__()
        # [EN] BUG FIX (real-world report: the app crashes/quits silently
        # on macOS, with the actual GtkGLArea widget suspected as the
        # cause). This widget's GL context was never given an explicit
        # required version -- meaning it relied entirely on GTK's own
        # DEFAULT context-negotiation behaviour, which on Linux/Mesa
        # tends to hand back something recent enough to run our
        # #version 330 shaders (geometry shaders, needed for the double/
        # triple bond rendering worked on earlier -- require OpenGL 3.2+
        # core) without ever being asked explicitly. macOS has NO
        # compatibility profile for OpenGL 3+ at all (confirmed via
        # Apple's own developer forums/documentation: only a "Legacy"
        # profile capped at the pre-3.2 feature set, or a strict Core
        # profile, nothing in between) -- so without requesting the
        # right version/profile up front, GTK's quartz backend may
        # negotiate a context our shaders can't actually use, or fail to
        # realize a usable context at all. set_required_version(3, 3)
        # asks for exactly what #version 330 needs, on every platform,
        # instead of leaving it to a default that happens to work on
        # Linux by coincidence rather than by being explicitly correct.
        if _IS_MACOS:
            self._gl_initialized = False
            self._gl_ctx = _OffscreenGLContext(width, height)
            self.connect("draw", self._draw_macos)
            self.connect("size-allocate", self._size_allocate_macos)
            # vismol_glcore.initialize() calls these on self.parent_widget
            # expecting the real Gtk.GLArea API -- no-op instance-level
            # shims here since the offscreen GLFW window is always
            # created with depth+alpha already (see _OffscreenGLContext).
            # Instance-level (not class-level) so Linux/Windows, which
            # never enters this branch, keeps using the real Gtk.GLArea
            # methods untouched.
            self.set_has_depth_buffer = lambda *args, **kwargs: None
            self.set_has_alpha        = lambda *args, **kwargs: None
        else:
            self.set_required_version(3, 3)
            self.connect("realize", self.initialize)
            self.connect("render", self.render)
            self.connect("resize", self.reshape)
        _patch_gl_line_width()
        self.connect("key-press-event", self.key_pressed)
        self.connect("key-release-event", self.key_released)
        self.connect("button-press-event", self.mouse_pressed)
        self.connect("button-release-event", self.mouse_released)
        self.connect("motion-notify-event", self.mouse_motion)
        self.connect("scroll-event", self.mouse_scroll)
        self.grab_focus()
        self.set_events(self.get_events() | Gdk.EventMask.SCROLL_MASK
                        | Gdk.EventMask.BUTTON_PRESS_MASK | Gdk.EventMask.BUTTON_RELEASE_MASK
                        | Gdk.EventMask.POINTER_MOTION_MASK | Gdk.EventMask.POINTER_MOTION_HINT_MASK
                        | Gdk.EventMask.KEY_PRESS_MASK | Gdk.EventMask.KEY_RELEASE_MASK)
        # self.vm_objects_list_store = Gtk.ListStore(bool,  # visible? 
        #                                            str,  # id
        #                                            str,  # name
        #                                            str,  # num of atoms
        #                                            str)  # num of frames
        self.vm_selection_modes_list_store = Gtk.ListStore(str)
        self.vm_session = vismol_session
        self.vm_glcore = VismolGLCore(self, vismol_session, width, height)
        self.glmenu_bg = None
        self.glmenu_sele = None
        self.glmenu_obj = None
        self.glmenu_pick = None
        self.filechooser = None
        self.selection_box_frame = None
    
    def initialize(self, widget):
        """ Enables the buffers and other charasteristics of the OpenGL context.
            sets the initial projection and view matrix
            
            self.flag -- Needed to only create one OpenGL program, otherwise a bunch of
                         programs will be created and use system resources. If the OpenGL
                         program will be changed change this value to True
        """
        if self.get_error() != None:
            # [EN] BUG FIX (real-world report: the app crashes/quits
            # silently on macOS, GtkGLArea suspected): this used to
            # report the error ONLY via dprint() -- which is SILENT
            # unless the EASYHYBRID_DEBUG=1 environment variable happens
            # to be set (see vismol/utils/debug.py) -- and then call
            # Gtk.main_quit() immediately, closing the whole application
            # with literally no visible explanation at all. Whatever the
            # underlying GL context failure actually was (e.g. exactly
            # the kind of version/profile mismatch set_required_version()
            # above is now meant to prevent) was completely invisible to
            # anyone not already running with that debug flag on --
            # indistinguishable from the app just silently dying for no
            # reason. Now always printed (not gated behind the debug
            # flag) before quitting, so at minimum the terminal shows
            # WHY, instead of nothing.
            error = self.get_error()
            print("FATAL: OpenGL context could not be created -- EasyHybrid cannot start.")
            print("  args:   ", error.args if error is not None else None)
            print("  code:   ", error.code if error is not None else None)
            print("  domain: ", error.domain if error is not None else None)
            print("  message:", error.message if error is not None else None)
            print("This usually means the GTK GLArea widget could not negotiate an OpenGL "
                  "3.3+ core context with your system's graphics driver. On macOS in "
                  "particular, this is a known area of difficulty for GTK3's own OpenGL "
                  "support -- see https://gitlab.gnome.org/GNOME/gtk (GTK3 quartz/macOS "
                  "backend) for the current state of this.")
            Gtk.main_quit()
            return
        self.vm_glcore.initialize()
    
    def reshape(self, widget, width, height):
        """ Resizing function, takes the widht and height of the widget
            and modifies the view in the camera acording to the new values
        
            Keyword arguments:
            widget -- The widget that is performing resizing
            width -- Actual width of the window
            height -- Actual height of the window
        """
        self.vm_glcore.resize_window(width, height)
        self.queue_draw()
    
    def render(self, area, context):
        """ This is the function that will be called everytime the window
            needs to be re-drawed.
        """
        self.vm_glcore.render()

    if _IS_MACOS:
        def _draw_macos(self, widget, cr):
            """ macOS-only draw path (Gtk.DrawingArea "draw"/Cairo signal),
                replacing the realize/render/resize trio used by
                Gtk.GLArea on Linux/Windows -- see the _IS_MACOS comment
                near the top of this file for why. Makes the offscreen
                GLFW context current, resizes it to match this widget's
                current allocation, renders the exact same scene via
                vm_glcore.render(), reads the finished frame back with
                glReadPixels, and blits it into the DrawingArea's Cairo
                context.
            """
            self._gl_ctx.make_current()
            if not self._gl_initialized:
                self._gl_initialized = True
                self.vm_glcore.initialize()

            width  = self.get_allocated_width()
            height = self.get_allocated_height()
            if width <= 0 or height <= 0:
                return True

            self._gl_ctx.resize(width, height)
            # [EN] BUG FIX (macOS trackpad zoom appeared to drift the
            # molecule sideways): glfw.set_window_size() above resizes the
            # hidden GLFW window/framebuffer, but -- unlike Gtk.GLArea on
            # Linux/Windows, which re-applies the GL viewport for you on
            # every resize -- it does NOT itself touch the GL viewport.
            # Without this call the viewport stayed stuck at whatever size
            # the hidden window had when _OffscreenGLContext was first
            # created (the widget's construction-time default, 640x420),
            # while vm_glcore.resize_window() (called from
            # _size_allocate_macos) kept the CAMERA's aspect ratio/width/
            # height/center_x/center_y correctly up to date for the real,
            # current window size. That mismatch between the projection
            # math (correct, current size) and the actual rasterization
            # viewport (stale, old size) is forgiving for rotate (angle-
            # only) but shows up as a sideways shift specifically while
            # dollying the camera in/out during zoom.
            glViewport(0, 0, width, height)
            self.vm_glcore.render()
            glFinish()
            data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)

            image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
            # OpenGL's origin is bottom-left, Cairo/screen origin is top-left.
            image = np.flip(image, axis=0)
            # Cairo's ARGB32 format is 4-byte-per-pixel B,G,R,A in memory
            # on little-endian (Apple Silicon is LE); background/geometry
            # in this view are always fully opaque, so straight vs.
            # premultiplied alpha makes no visible difference here.
            bgra = np.ascontiguousarray(image[:, :, [2, 1, 0, 3]])

            surface = cairo.ImageSurface.create_for_data(
                bytearray(bgra.tobytes()), cairo.FORMAT_ARGB32, width, height, width * 4)
            cr.set_source_surface(surface, 0, 0)
            cr.paint()
            return True

        def _size_allocate_macos(self, widget, allocation):
            """ macOS-only resize path (Gtk.DrawingArea "size-allocate"
                signal), equivalent to reshape() used by Gtk.GLArea on
                Linux/Windows -- updates the camera/projection to match
                the new size and asks for a redraw.
            """
            self.vm_glcore.resize_window(allocation.width, allocation.height)
            self.queue_draw()

    def mouse_pressed(self, widget, event):
        """ Function doc """
        if _IS_MACOS:
            self.vm_glcore.mouse_pressed(event.button, event.x, event.y, event.state)
        else:
            self.vm_glcore.mouse_pressed(event.button, event.x, event.y)
    
    def mouse_released(self, widget, event):
        """ Function doc """
        if _IS_MACOS:
            self.vm_glcore.mouse_released(event.button, event.x, event.y, event.state)
        else:
            self.vm_glcore.mouse_released(event.button, event.x, event.y)
    
    def mouse_motion(self, widget, event):
        """ Function doc """
        if _IS_MACOS:
            self.vm_glcore.mouse_motion(event.x, event.y, event.state)
        else:
            self.vm_glcore.mouse_motion(event.x, event.y)
    
    def mouse_scroll(self, widget, event):
        """ Function doc
        """
        if event.direction == Gdk.ScrollDirection.UP:
            self.vm_glcore.mouse_scroll(1)
        if event.direction == Gdk.ScrollDirection.DOWN:
            self.vm_glcore.mouse_scroll(-1)
    
    def _build_glmenu(self, bg_menu=None, sele_menu=None, obj_menu=None, pick_menu=None):
        """ Function doc """
        if bg_menu is not None:
            self.glmenu_bg = Gtk.Menu()
            self.glmenu_bg_toplabel = Gtk.MenuItem(label="background")
            self._build_glmenu_from_dicts(bg_menu, self.glmenu_bg)
            self.glmenu_bg.show_all()
        else:
            self.glmenu_bg = None
        
        if sele_menu is not None:
            self.glmenu_sele = Gtk.Menu()
            self.glmenu_sele_toplabel = Gtk.MenuItem(label="selection")
            self._build_glmenu_from_dicts(sele_menu, self.glmenu_sele)
            self.glmenu_sele.show_all()
        else:
            self.glmenu_sele = None
        
        if pick_menu is not None:
            self.glmenu_pick = Gtk.Menu()
            self.glmenu_pick_toplabel = Gtk.MenuItem(label="picking")
            self.glmenu_pick.append(self.glmenu_pick_toplabel)
            self._build_glmenu_from_dicts(pick_menu, self.glmenu_pick)
            self.glmenu_pick.show_all()
        else:
            self.glmenu_pick = None
        
        if obj_menu is not None:
            self.glmenu_obj = Gtk.Menu()
            self.glmenu_obj_toplabel = Gtk.MenuItem(label="object")
            self.glmenu_obj.append(self.glmenu_obj_toplabel)
            self._build_glmenu_from_dicts(obj_menu, self.glmenu_obj)
            self.glmenu_obj.show_all()
        else:
            self.glmenu_obj = None
     
    def open_file(self, widget):
        """ Function doc """
        if self.filechooser is None:
            self.filechooser = FileChooser()
        filename = self.filechooser.open()
        self.vm_session.load_molecule(filename)
    
    def key_pressed(self, widget, event):
        """ The key_pressed function serves, as the names states, to catch
            events in the keyboard, e.g. letter "l" pressed, "backslash"
            pressed. Note that there is a difference between "A" and "a".
            Here I use a specific handler for each key pressed after
            discarding the CONTROL, ALT and SHIFT keys pressed (usefull
            for customized actions) and maintained, i.e. it"s the same as
            using Ctrl+Z to undo an action.
        """
        try:
            func = getattr(self, "_pressed_" + Gdk.keyval_name(event.keyval))
            func()
        except AttributeError as ae:
            logger.debug("Press key {} has not been assigned to a handler "\
                         "yet".format(Gdk.keyval_name(event.keyval)))
    
    def key_released(self, widget, event):
        """ Used to indicates a key has been released.
        """
        try:
            func = getattr(self, "_released_" + Gdk.keyval_name(event.keyval))
            func()
        except AttributeError as ae:
            logger.debug("Release key {} has not been assigned to a handler "\
                         "yet".format(Gdk.keyval_name(event.keyval)))
    
    def _pressed_Escape(self):
        """ Function doc """
        self.quit()
    
    def _pressed_Right(self):
        """ Function doc """
        self.vm_session.forward_frame()
        self.queue_draw()

    def _released_Right(self):
        """ Function doc """
        pass
    
    def _pressed_Left(self):
        """ Function doc """
        self.vm_session.reverse_frame()
        self.queue_draw()
    
    def _released_Left(self):
        """ Function doc """
        pass
    
    def _pressed_Control_L(self):
        """ Function doc """
        self.vm_glcore.ctrl = True
    
    def _released_Control_L(self):
        """ Function doc """
        self.vm_glcore.ctrl = False
    
    def _pressed_Shift_L(self):
        """ Function doc """
        self.vm_glcore.shift = True
    
    def _released_Shift_L(self):
        """ Function doc """
        self.vm_glcore.shift = False

    if _IS_MACOS:
        # [EN] macOS trackpad fix: camera panning is bound to a
        # middle-mouse-button drag (vismol_glcore.py), but MacBook
        # trackpads have no middle button and no default gesture that
        # emulates one -- panning was effectively unreachable from a
        # trackpad. Holding Cmd while right-dragging (Cmd + two-finger-
        # drag on a trackpad) now also triggers pan (see
        # VismolGLCore.mouse_pressed's "and not self.cmd"/"or (... and
        # self.cmd ...)" gating), matching PyMOL's own Cmd+Right =
        # translate convention on macOS. Which keyval GTK reports for the
        # Cmd key can vary by GTK version/backend/keymap (only Meta_L was
        # actually observed on the tested setup); all four are wired as
        # cheap insurance. Gated to macOS only -- Meta_L/Super_L are real,
        # occasionally-bound X11 keysyms on Linux, so leaving these
        # handlers undefined there guarantees zero behavior change.
        def _pressed_Meta_L(self):
            """ Function doc """
            self.vm_glcore.cmd = True

        def _released_Meta_L(self):
            """ Function doc """
            self.vm_glcore.cmd = False

        _pressed_Meta_R  = _pressed_Meta_L
        _released_Meta_R = _released_Meta_L
        _pressed_Super_L  = _pressed_Meta_L
        _released_Super_L = _released_Meta_L
        _pressed_Super_R  = _pressed_Meta_L
        _released_Super_R = _released_Meta_L

    # [EN] Builder keyboard shortcuts ('a'/'d'/'b') -- only act while
    # Builder editing mode is on (builder_atom_mode), so these letter
    # keys don't hijack anything when the Builder isn't in use (e.g. if
    # 'a'/'b'/'d' end up wanted for some other, non-Builder shortcut
    # later, or are just typed for an unrelated reason). See
    # gui/windows/builder/click_mode.py for what each one does.
    def _pressed_a(self):
        """ Builder: switch to the "add atom" tool (plain click places a
        new atom) -- also the default tool whenever Builder mode is
        first turned on. """
        if getattr ( self.vm_session, "builder_atom_mode", False ):
            from gui.windows.builder.click_mode import set_tool
            set_tool ( self.vm_session, "add" )
            dprint ( "Builder: tool = add atom" )

    def _pressed_d(self):
        """ Builder: switch to the "delete atom" tool (plain click
        removes the clicked atom). """
        if getattr ( self.vm_session, "builder_atom_mode", False ):
            from gui.windows.builder.click_mode import set_tool
            set_tool ( self.vm_session, "delete" )
            dprint ( "Builder: tool = delete atom" )

    def _pressed_b(self):
        """ Builder: one-shot action -- adds a bond between the two
        atoms currently selected (shift-click two atoms first; see
        click_mode.handle_bond_shortcut() for the exact requirements). """
        if getattr ( self.vm_session, "builder_atom_mode", False ):
            from gui.windows.builder.click_mode import handle_bond_shortcut
            msg = handle_bond_shortcut ( self.vm_session,
                    bond_order = getattr ( self.vm_session, "builder_bond_order", 1 ),
                    aromatic   = getattr ( self.vm_session, "builder_bond_aromatic", False ) )
            dprint ( "Builder: {}".format ( msg ) )
            self.queue_draw ( )
    
    def _selection_type_picking(self, widget):
        if self.selection_box_frame:
            self.selection_box_frame.change_toggle_button_selecting_mode_status(True)
        else:
            self.vm_session.picking_selection_mode = True
        self.queue_draw()
    
    def _selection_type_viewing(self, widget):
        if self.selection_box_frame:
            self.selection_box_frame.change_toggle_button_selecting_mode_status(False)
        else:
            self.vm_session.picking_selection_mode = False
        self.queue_draw()
    
    def quit(self):
        logger.info("Thanks for using our software :). Quitting Vismol.")
        Gtk.main_quit()
    
    def _viewing_selection_mode_atom(self, widget):
        self.vm_session.viewing_selection_mode(sel_type="atom")
    
    def _viewing_selection_mode_residue(self, widget):
        self.vm_session.viewing_selection_mode(sel_type="residue")
    
    def _viewing_selection_mode_chain(self, widget):
        self.vm_session.viewing_selection_mode(sel_type="chain")
    
    def menu_show_dots(self, widget):
        self.vm_session.show_or_hide(rep_type="dots", show=True)
    
    def menu_hide_dots(self, widget):
        self.vm_session.show_or_hide(rep_type="dots", show=False)
    
    def menu_show_lines(self, widget):
        self.vm_session.show_or_hide(rep_type="lines", show=True)
    
    def menu_hide_lines(self, widget):
        self.vm_session.show_or_hide(rep_type="lines", show=False)
    
    def menu_show_nonbonded(self, widget):
        self.vm_session.show_or_hide(rep_type="nonbonded", show=True)
    
    def menu_hide_nonbonded(self, widget):
        self.vm_session.show_or_hide(rep_type="nonbonded", show=False)
    
    def menu_show_impostor(self, widget):
        self.vm_session.show_or_hide(rep_type="impostor", show=True)
    
    def menu_hide_impostor(self, widget):
        self.vm_session.show_or_hide(rep_type="impostor", show=False)
    
    def menu_show_spheres(self, widget):
        self.vm_session.show_or_hide(rep_type="spheres", show=True)
    
    def menu_hide_spheres(self, widget):
        self.vm_session.show_or_hide(rep_type="spheres", show=False)
    
    def menu_show_sticks(self, widget):
        self.vm_session.show_or_hide(rep_type="sticks", show=True)
    
    def menu_hide_sticks(self, widget):
        self.vm_session.show_or_hide(rep_type="sticks", show=False)
    
    def invert_selection(self, widget):
        self.vm_session.selections[self.vm_session.current_selection].invert_selection()
    
    def insert_glmenu(self, bg_menu=None, sele_menu=None, obj_menu=None, pick_menu=None):
        """ Function doc """
        if bg_menu is None:
            """ Standard Bg Menu"""
            bg_menu = {"Open File": ["MenuItem", self.open_file],
                       "separator": ["separator", None],
                       "Selection Mode": ["submenu",
                                            {"by atom": ["MenuItem", self._viewing_selection_mode_atom],
                                             "by residue": ["MenuItem", self._viewing_selection_mode_residue],
                                             "by chain": ["MenuItem", self._viewing_selection_mode_chain],
                                            }
                                         ],
                       "Selection Type": ["submenu",
                                            {"viewing": ["MenuItem", self._selection_type_viewing],
                                             "picking": ["MenuItem", self._selection_type_picking],
                                            }
                                         ],
                       "separator": ["separator", None],
                       "Quit": ["MenuItem", self.quit],
                      }
        
        if sele_menu is None:
            """ Standard Sele Menu """
            sele_menu = {"Show": ["submenu",
                                    {"dots": ["MenuItem", self.menu_show_dots],
                                     "lines": ["MenuItem", self.menu_show_lines],
                                     "nonbonded": ["MenuItem", self.menu_show_nonbonded],
                                     "sticks": ["MenuItem", self.menu_show_sticks],
                                     "impostor": ["MenuItem", self.menu_show_impostor],
                                     "spheres": ["MenuItem", self.menu_show_spheres],
                                    }
                                 ],
                         "Hide": ["submenu",
                                    {"dots": ["MenuItem", self.menu_hide_dots],
                                     "lines": ["MenuItem", self.menu_hide_lines],
                                     "nonbonded": ["MenuItem", self.menu_hide_nonbonded],
                                     "sticks": ["MenuItem", self.menu_hide_sticks],
                                     "impostor": ["MenuItem", self.menu_hide_impostor],
                                     "spheres": ["MenuItem", self.menu_hide_spheres],
                                    }
                                 ],
                         "separator":["separator", None],
                         "Invert Selection": ["MenuItem", self.invert_selection],
                        }
        
        if obj_menu is None:
            """ Standard Obj Menu"""
            obj_menu = {"Show": ["submenu",
                                    {"dots": ["MenuItem", None],
                                     "lines": ["MenuItem", None],
                                     "nonbonded": ["MenuItem", None],
                                    }
                                ],
                        "Hide": ["submenu",
                                    {"dots": ["MenuItem", None],
                                     "lines": ["MenuItem", None],
                                     "nonbonded": ["MenuItem", None],
                                    }
                                ],
                        "separator":["separator", None],
                        "label": ["submenu",
                                    {"Atom": ["submenu",
                                                {"index": ["MenuItem", None],
                                                 "name": ["MenuItem", None],
                                                 "residue": ["MenuItem", None],
                                                 "chain": ["MenuItem", None],
                                                 }
                                             ],
                                     "Residue": ["submenu",
                                                    {"index": ["MenuItem", None],
                                                     "name": ["MenuItem", None],
                                                     "chain": ["MenuItem", None],
                                                     }
                                                ],
                                     "Chain": ["submenu",
                                                {"name": ["MenuItem", None],
                                                 }
                                              ],
                                    },
                                 ],
                        }
        
        if pick_menu is None:
            """ Standard Sele Menu """
            pick_menu = {"Show": ["submenu",
                                    {"dots": ["MenuItem", None],
                                     "lines": ["MenuItem", None],
                                     "nonbonded": ["MenuItem", None],
                                    }
                                  ],
                         "Hide": ["submenu",
                                    {"dots": ["MenuItem", None],
                                     "lines": ["MenuItem", None],
                                     "nonbonded": ["MenuItem", None],
                                    }
                                  ],
                        }
        
        self._build_glmenu(bg_menu=bg_menu, sele_menu=sele_menu, obj_menu=obj_menu, pick_menu=pick_menu)
    
    def show_gl_menu(self, signals=None, menu_type=None, info=None):
        """ Function doc """
        if menu_type == "bg_menu":
            if self.glmenu_bg:
                self.glmenu_bg.popup(None, None, None, None, 0, 0)
        
        if menu_type == "sele_menu":
            if self.glmenu_sele:
                self.glmenu_sele.popup(None, None, None, None, 0, 0)
        
        if menu_type == "pick_menu":
            if self.glmenu_pick:
                self.glmenu_pick.popup(None, None, None, None, 0, 0)
        
        if menu_type == "obj_menu":
            if self.glmenu_obj:
                self.glmenu_obj_toplabel.set_label(info)
                self.glmenu_obj.popup(None, None, None, None, 0, 0)
    
    def _build_submenus_from_dicts(self, menu_dict):
        """ Function doc """
        menu = Gtk.Menu()
        for key in menu_dict:
            mitem = Gtk.MenuItem(key)
            if menu_dict[key][0] == "submenu":
                menu2 = self._build_submenus_from_dicts(menu_dict[key][1])
                mitem.set_submenu(menu2)
            elif menu_dict[key][0] == "separator":
                mitem = Gtk.SeparatorMenuItem()
            else:
                if menu_dict[key][1] != None:
                    mitem.connect("activate", menu_dict[key][1])
                else:
                    pass
            menu.append(mitem)
        return menu
    
    def _build_glmenu_from_dicts(self, menu_dict, glMenu):
        """ Function doc """
        for key in menu_dict:
            mitem = Gtk.MenuItem(label=key)
            if menu_dict[key][0] == "submenu":
                menu2 = self._build_submenus_from_dicts(menu_dict[key][1])
                mitem.set_submenu(menu2)
            elif menu_dict[key][0] == "separator":
                mitem = Gtk.SeparatorMenuItem()
            else:
                if menu_dict[key][1] != None:
                    mitem.connect("activate", menu_dict[key][1])
                else:
                    pass
            glMenu.append(mitem)
    
    def capture_screenshot(self, scale_factor=1):
        """ Renders a fresh frame and reads it back from the OpenGL
            framebuffer as a numpy RGBA array (height, width, 4), ready
            to be saved or further processed (e.g. the cartoon filter in
            PreviewWindow).

            Keyword arguments:
            scale_factor -- render at this multiple of the widget's
                             current on-screen resolution (e.g. 2 or 3
                             for a 2x/3x resolution export). Rendering at
                             a higher resolution - rather than upscaling
                             the captured image afterwards - actually
                             produces finer geometry (thinner/smoother
                             lines, circles, text), not just a blurrier
                             enlargement of the same pixels. Default 1
                             captures at the widget's current size.

            Returns:
            image -- np.uint8 array of shape (height*scale_factor,
                     width*scale_factor, 4), or None if the GL context
                     could not be made current or the offscreen
                     framebuffer (when scale_factor > 1) failed.
        """
        if _IS_MACOS:
            self._gl_ctx.make_current()
        else:
            self.make_current()

            if self.get_error() is not None:
                dprint("Error in OpenGL context")
                return None

        if scale_factor != 1:
            # vm_glcore.render_to_image() handles the offscreen
            # framebuffer at the higher resolution, including rendering
            # into it and restoring every bit of state afterwards.
            return self.vm_glcore.render_to_image(scale_factor)

        # Render a fresh frame synchronously into this widget's GL
        # context right before reading pixels back, so the capture
        # reflects the current camera/scene state instead of whatever
        # was left in the backbuffer from a previous (possibly partial)
        # draw cycle.
        self.vm_glcore.render()

        width = self.get_allocated_width()
        height = self.get_allocated_height()

        # Mesma garantia de sincronizacao aplicada em vm_glcore.render_to_image
        # -- garante que o desenho terminou de verdade na GPU antes da leitura.
        glFinish()
        data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)

        # Converter para numpy
        image = np.frombuffer(data, dtype=np.uint8)
        image = image.reshape((height, width, 4))

        # Inverter verticalmente (OpenGL tem origem no canto inferior
        # esquerdo; imagens/PNG tem origem no canto superior esquerdo)
        image = np.flip(image, axis=0)
        # np.flip returns a view with negative strides, which PIL/some
        # encoders can choke on - make it a normal contiguous array.
        image = np.ascontiguousarray(image)
        return image

    def save_image(self, filename, scale_factor=1):
        """ Captures the current OpenGL view and saves it as a PNG file.

            Keyword arguments:
            filename -- path (including ".png" extension) where the
                        screenshot will be saved.
            scale_factor -- see capture_screenshot(); 2 or 3 for a
                             2x/3x resolution export, 1 (default) for
                             the widget's current on-screen resolution.

            Returns:
            True on success, False if the capture failed (e.g. invalid
            GL context or offscreen framebuffer error).
        """
        image = self.capture_screenshot(scale_factor=scale_factor)
        if image is None:
            return False

        # Note: this keeps the alpha channel as read from the framebuffer.
        # The GLArea is created with set_has_alpha(True) (see
        # VismolGLCore.initialize()), so background pixels may come back
        # non-opaque depending on the configured background_color alpha.
        # That's preserved here rather than silently flattened, since a
        # transparent-background PNG may be exactly what's wanted (e.g.
        # to drop the molecule into another image/slide).
        img = Image.fromarray(image, mode="RGBA")
        img.save(filename)
        dprint("Imagem salva em {}".format(filename))
        return True

    def open_export_preview(self):
        """ Captures the current OpenGL view (at screen resolution, for
            a cheap live preview) and opens the interactive Preview/Export
            window (PreviewWindow). From there the user can toggle/tweak
            the cartoon filter, save/load named style presets, and export
            the final PNG at a chosen resolution multiplier - which
            re-captures the scene at full quality rather than upscaling
            this preview image.
        """
        image = self.capture_screenshot()
        if image is None:
            return
        open_preview(image, glwidget=self)



# ---------------------------------------------------------------------------
# Pure image-processing / preset logic (no GTK), so it's testable on its
# own and reusable both for the cheap live preview and the final export.
# ---------------------------------------------------------------------------

FILTER_DEFAULTS = {
    "blur": 1.2,
    "colors": 16,
    "edge_threshold": 50,
    "blend": 0.8,
}
# Kept as an alias for backwards compatibility with any saved presets
# that still reference the old name.
CARTOON_FILTER_DEFAULTS = FILTER_DEFAULTS

FILTER_MODES = ("none", "outline", "cartoon")

RENDER_PRESETS_DIR = os.path.join(os.environ.get("HOME", "."), ".VisMol", "render_presets")


def apply_outline_filter(img, params, posterize):
    """ PyMOL-style outline post-processing. Mirrors PyMOL's
        ray_trace_mode: mode 1 ("normal color + black outline") is
        posterize=False here, mode 3 ("quantized color + black outline",
        the more cartoon-ish look) is posterize=True.

        Detecting the outline from the source image directly (rather
        than from PyMOL's real depth/normal buffers, which this 2D
        post-process doesn't have access to) means smooth shading
        gradients on a sphere can register as faint false edges. Two
        things keep that under control without needing depth data:
        blurring before edge detection (smooths out the gradual
        shading gradient while keeping the sharp true silhouette), and
        a high enough edge_threshold (the true silhouette's contrast is
        much higher than the residual shading gradient's). Detecting
        edges from the posterized image instead of the blurred original
        was tried and rejected: posterization itself splits each
        sphere's shading into a few flat bands, which then show up as
        extra concentric false-edge rings - worse than the residual
        gradient noise it was meant to fix.

        Keyword arguments:
        img       -- PIL.Image (RGB or RGBA) to process.
        params    -- dict with keys "blur", "colors", "edge_threshold",
                     "blend" (see FILTER_DEFAULTS for ranges/defaults).
                     "colors" is only used when posterize is True.
        posterize -- if True, quantize colors first (PyMOL mode 3 /
                     cartoon look). If False, keep full original colors
                     and shading, just add the outline (PyMOL mode 1).

        Returns:
        A new PIL.Image (RGB) with the effect applied.
    """
    blur = float(params.get("blur", FILTER_DEFAULTS["blur"]))
    colors = int(params.get("colors", FILTER_DEFAULTS["colors"]))
    edge_th = int(params.get("edge_threshold", FILTER_DEFAULTS["edge_threshold"]))
    blend_val = float(params.get("blend", FILTER_DEFAULTS["blend"]))

    base = img.convert("RGB")
    smooth = base.filter(ImageFilter.GaussianBlur(blur))

    if posterize:
        # MAXCOVERAGE preserves saturated, distinct colors much better
        # than the default MEDIANCUT method, which tends to compromise
        # between a sphere's lit/shaded gradient and washes the result
        # out towards gray. dither=0 (Dither.NONE) avoids a visible
        # checkerboard-like speckle pattern that Floyd-Steinberg
        # dithering (the default) leaves in smooth gradient areas like
        # a sphere's shading.
        colored = smooth.quantize(colors=colors, method=Image.MAXCOVERAGE, dither=0).convert("RGB")
    else:
        colored = base

    # Edge mask, detected from the blurred (not posterized - see
    # docstring) image. FIND_EDGES produces HIGH values exactly where
    # there's an edge and LOW (near 0) on flat areas. We want a mask
    # where EDGE pixels are BLACK (0, so they darken via multiply below)
    # and everything else is WHITE (255, neutral under multiply) -
    # getting this polarity backwards (e.g. inverting before
    # thresholding) makes the flat background end up white instead of
    # the edges, which then washes the entire image towards gray in the
    # final blend instead of just outlining the edges.
    edges_raw = smooth.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_mask = edges_raw.point(lambda x: 0 if x >= edge_th else 255).convert("RGB")

    # multiply only darkens where the mask is black (the detected
    # edges), leaving flat/background areas exactly as they were in
    # `colored`. blend_val then controls how strong that outline is,
    # from 0 (no outline) to 1 (fully black outline where detected).
    outlined = ImageChops.multiply(colored, edge_mask)
    return Image.blend(colored, outlined, blend_val)


def apply_cartoon_filter(img, params):
    """ Backwards-compatible alias for apply_outline_filter(img, params,
        posterize=True) - the "cartoon" look (PyMOL ray_trace_mode 3).
    """
    return apply_outline_filter(img, params, posterize=True)


class RenderStylePreset:
    """ A named, JSON-serializable bundle of render/filter parameters.

        This is what lets a user style one frame in the preview window,
        save the preset by name, and later apply that exact same style
        to other frames/screenshots without redoing the sliders by hand.
    """

    def __init__(self, name="default", filter_mode="none", filter_params=None):
        """
        filter_mode -- one of FILTER_MODES:
                       "none"    - no post-processing (raw screenshot).
                       "outline" - PyMOL ray_trace_mode 1 style: original
                                   colors/shading + a black outline.
                       "cartoon" - PyMOL ray_trace_mode 3 style: quantized
                                   ("posterized") colors + a black outline.
        """
        self.name = name
        self.filter_mode = filter_mode if filter_mode in FILTER_MODES else "none"
        self.filter_params = dict(FILTER_DEFAULTS)
        if filter_params:
            self.filter_params.update(filter_params)

    @property
    def cartoon_enabled(self):
        """ Backwards-compatible view of the old boolean flag. """
        return self.filter_mode == "cartoon"

    def to_dict(self):
        return {
            "name": self.name,
            "filter_mode": self.filter_mode,
            "filter_params": self.filter_params,
        }

    @classmethod
    def from_dict(cls, data):
        # Back-compat: presets saved before filter_mode existed only
        # had a "cartoon_enabled" boolean and "cartoon_params".
        if "filter_mode" not in data and "cartoon_enabled" in data:
            mode = "cartoon" if data.get("cartoon_enabled") else "none"
            params = data.get("cartoon_params")
        else:
            mode = data.get("filter_mode", "none")
            params = data.get("filter_params")
        return cls(name=data.get("name", "default"), filter_mode=mode, filter_params=params)

    def apply(self, img):
        """ Returns img with this preset's effect applied (or img
            untouched, converted to RGB, if filter_mode is "none").
        """
        if self.filter_mode == "none":
            return img.convert("RGB")
        return apply_outline_filter(img, self.filter_params,
                                     posterize=(self.filter_mode == "cartoon"))

    @staticmethod
    def _safe_filename(name):
        keep = "-_ "
        cleaned = "".join(c for c in name if c.isalnum() or c in keep).strip()
        return (cleaned or "preset") + ".json"

    def save(self, presets_dir=RENDER_PRESETS_DIR):
        """ Saves this preset as <presets_dir>/<name>.json (name is
            sanitized for use as a filename). Creates presets_dir if it
            doesn't exist yet.
        """
        os.makedirs(presets_dir, exist_ok=True)
        path = os.path.join(presets_dir, self._safe_filename(self.name))
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @staticmethod
    def list_presets(presets_dir=RENDER_PRESETS_DIR):
        """ Returns a sorted list of preset names available in
            presets_dir (the .json extension is stripped). Returns an
            empty list if the directory doesn't exist yet.
        """
        if not os.path.isdir(presets_dir):
            return []
        names = [os.path.splitext(f)[0] for f in os.listdir(presets_dir)
                  if f.endswith(".json")]
        return sorted(names)


# ---------------------------------------------------------------------------
# GTK preview window
# ---------------------------------------------------------------------------

class PreviewWindow(Gtk.Window):
    """ Shows a small live preview of the current OpenGL view with an
        optional cartoon filter, lets the user tweak the filter with
        sliders, save/load named style presets, and export the result at
        a chosen resolution.

        Unlike a quick "process the already-captured image" approach,
        Export re-captures the scene through glwidget at the requested
        scale_factor (via VismolGTKWidget.capture_screenshot /
        VismolGLCore.render_to_image) so the exported PNG is rendered at
        full quality, not an upscaled/blurred copy of the small preview.
    """

    def __init__(self, image_array, glwidget):
        super().__init__(title="Preview / Export")
        self.glwidget = glwidget

        self.set_default_size(640, 420)
        # Independent top-level window, not a nested mainloop: closing it
        # only destroys this window, it doesn't quit the application.
        self.connect("destroy", lambda *_: None)

        # imagem original (PIL) - this is just the cheap on-screen-res
        # capture used for the live preview, not what gets exported.
        self.original = Image.fromarray(image_array)
        #self.preview_base = self.original.resize(
        #    (max(1, self.original.width // 4), max(1, self.original.height // 4))
        #)
        self.preview_base = self.original

        # layout principal
        self.general_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.general_box.set_border_width(10)
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        self.general_box.pack_start(box, True, True, 0)
        
        self.add(self.general_box)

        self.image_widget = Gtk.Image()
        box.pack_start(self.image_widget, True, True, 0)

        controls = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        box.pack_start(controls, False, False, 0)

        # modo de filtro: nenhum, contorno (cor original, estilo PyMOL
        # ray_trace_mode 1), ou cartoon (cor posterizada + contorno,
        # estilo PyMOL ray_trace_mode 3).
        mode_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        
        self.btn_refresh = Gtk.Button(label="Refresh")
        self.btn_refresh.connect("clicked", self.on_refresh_image)
        
        controls.pack_start(self.btn_refresh, False, False, 0)
        
        mode_box.pack_start(Gtk.Label(label="Style:"), False, False, 0)
        
        mode_box.pack_start(Gtk.Label(label="Style:"), False, False, 0)
        self.mode_combo = Gtk.ComboBoxText()
        self._mode_labels = [
            #("none", "Sem efeito"),
            #("outline", "Contorno (cor original)"),
            #("cartoon", "Cartoon (posterizado)"),
            
            ("none"   , "No effect"), 
            ("outline", 'Outline (original color)'), 
            ("cartoon", "Cartoon (posterized)"),
        ]
        for _, label in self._mode_labels:
            self.mode_combo.append_text(label)
        self.mode_combo.set_active(0)
        self.mode_combo.connect("changed", self.on_change)
        mode_box.pack_start(self.mode_combo, False, False, 0)
        controls.pack_start(mode_box, False, False, 0)

        self.blur = self._create_slider("Blur", 0.0, 5.0,
                                         FILTER_DEFAULTS["blur"], controls)
        self.colors = self._create_slider("Color (cartoon mode only)", 2, 64,
                                           FILTER_DEFAULTS["colors"], controls)
        self.edge = self._create_slider("Borders", 0, 255,
                                         FILTER_DEFAULTS["edge_threshold"], controls)
        self.blend = self._create_slider("Contour Intensity", 0.0, 1.0,
                                          FILTER_DEFAULTS["blend"], controls)

        controls.pack_start(Gtk.Separator(), False, False, 4)

        # --- presets: salvar/carregar o conjunto de parâmetros acima ---
        preset_label = Gtk.Label(label="Style Preset:")
        preset_label.set_halign(Gtk.Align.START)
        controls.pack_start(preset_label, False, False, 0)

        self.preset_combo = Gtk.ComboBoxText()
        self._refresh_preset_list()
        controls.pack_start(self.preset_combo, False, False, 0)

        preset_btn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        load_btn = Gtk.Button(label="load")
        load_btn.connect("clicked", self.on_load_preset)
        preset_btn_box.pack_start(load_btn, True, True, 0)
        save_btn = Gtk.Button(label="Save as...")
        save_btn.connect("clicked", self.on_save_preset)
        preset_btn_box.pack_start(save_btn, True, True, 0)
        controls.pack_start(preset_btn_box, False, False, 0)

        controls.pack_start(Gtk.Separator(), False, False, 4)

        # --- export: escala + caminho do arquivo ---
        scale_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        scale_box.pack_start(Gtk.Label(label="Resolution:"), False, False, 0)
        self.scale_combo = Gtk.ComboBoxText()
        for option in ["1x (screen)", "2x", "3x", "4x"]:
            self.scale_combo.append_text(option)
        self.scale_combo.set_active(1)  # default 2x, já que é o motivo de ter essa janela
        scale_box.pack_start(self.scale_combo, False, False, 0)
        controls.pack_start(scale_box, False, False, 0)

        export_btn = Gtk.Button(label="Export PNG...")
        export_btn.connect("clicked", self.on_export)
        controls.pack_start(export_btn, False, False, 0)

        self.status_label = Gtk.Label(label="")
        self.status_label.set_halign(Gtk.Align.START)
        self.general_box.pack_start(self.status_label, False, False, 0)

        self.update_preview()

    def _create_slider(self, label, minv, maxv, default, parent):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        parent.pack_start(box, False, False, 0)

        lbl = Gtk.Label(label=label)
        lbl.set_halign(Gtk.Align.START)
        box.pack_start(lbl, False, False, 0)

        adj = Gtk.Adjustment(default, minv, maxv, 0.1, 1, 0)
        scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=adj)
        scale.connect("value-changed", self.on_change)
        box.pack_start(scale, False, False, 0)
        return scale

    def _current_preset(self, name="default"):
        """ Builds a RenderStylePreset from the current state of every
            control in the window.
        """
        mode_index = self.mode_combo.get_active()
        mode = self._mode_labels[mode_index][0] if mode_index >= 0 else "none"
        return RenderStylePreset(
            name=name,
            filter_mode=mode,
            filter_params={
                "blur": self.blur.get_value(),
                "colors": int(self.colors.get_value()),
                "edge_threshold": int(self.edge.get_value()),
                "blend": self.blend.get_value(),
            },
        )

    def _apply_preset_to_controls(self, preset):
        """ Pushes a RenderStylePreset's values into the sliders/mode
            combo (the reverse of _current_preset), then refreshes the
            preview.
        """
        mode_to_index = {mode: i for i, (mode, _) in enumerate(self._mode_labels)}
        self.mode_combo.set_active(mode_to_index.get(preset.filter_mode, 0))
        self.blur.set_value(preset.filter_params.get("blur", FILTER_DEFAULTS["blur"]))
        self.colors.set_value(preset.filter_params.get("colors", FILTER_DEFAULTS["colors"]))
        self.edge.set_value(preset.filter_params.get("edge_threshold", FILTER_DEFAULTS["edge_threshold"]))
        self.blend.set_value(preset.filter_params.get("blend", FILTER_DEFAULTS["blend"]))
        self.update_preview()

    def update_preview(self):
        preset = self._current_preset()
        img = preset.apply(self.preview_base)

        data = np.array(img)
        height, width, _ = data.shape

        pixbuf = GdkPixbuf.Pixbuf.new_from_data(
            data.tobytes(),
            GdkPixbuf.Colorspace.RGB,
            False,
            8,
            width,
            height,
            width * 3
        )
        # keep a reference alive for the lifetime of the pixbuf - GTK
        # doesn't copy the buffer, only holds this Python object's data
        self._preview_pixbuf_data = data
        self.image_widget.set_from_pixbuf(pixbuf)

    def on_change(self, widget):
        self.update_preview()

    def _refresh_preset_list(self):
        self.preset_combo.remove_all()
        for name in RenderStylePreset.list_presets():
            self.preset_combo.append_text(name)

    def on_refresh_image (self, widget):
        """ Function doc """
        dprint('Refresh image')
        image = self.glwidget.capture_screenshot()
        self.preview_base =  Image.fromarray(image)
        self.update_preview()
        
    def on_save_preset(self, widget):
        dialog = Gtk.Dialog(title="Save Preset", transient_for=self, modal=True)
        dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                            Gtk.STOCK_SAVE, Gtk.ResponseType.OK)
        entry = Gtk.Entry()
        entry.set_text("my style")
        entry.set_activates_default(True)
        dialog.get_content_area().pack_start(Gtk.Label(label="Preset name:"), False, False, 6)
        dialog.get_content_area().pack_start(entry, False, False, 6)
        dialog.set_default_response(Gtk.ResponseType.OK)
        dialog.show_all()
        response = dialog.run()
        name = entry.get_text().strip()
        dialog.destroy()

        if response == Gtk.ResponseType.OK and name:
            preset = self._current_preset(name=name)
            path = preset.save()
            self._refresh_preset_list()
            self.status_label.set_text("Preset salvo: {}".format(path))

    def on_load_preset(self, widget):
        name = self.preset_combo.get_active_text()
        if not name:
            self.status_label.set_text("No preset selected.")
            return
        path = os.path.join(RENDER_PRESETS_DIR, RenderStylePreset._safe_filename(name))
        try:
            preset = RenderStylePreset.load(path)
        except (OSError, json.JSONDecodeError) as e:
            self.status_label.set_text("Failed to load preset: {}".format(e))
            return
        self._apply_preset_to_controls(preset)
        self.status_label.set_text("Preset loaded: {}".format(name))

    def on_export_old(self, widget):
        scale_text = self.scale_combo.get_active_text()
        scale_factor = int(scale_text[0])

        dialog = Gtk.FileChooserDialog(
            title="Export PNG", parent=self, action=Gtk.FileChooserAction.SAVE)
        dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                           Gtk.STOCK_SAVE, Gtk.ResponseType.OK)
        file_filter = Gtk.FileFilter()
        file_filter.set_name("PNG image (*.png)")
        file_filter.add_pattern("*.png")
        dialog.add_filter(file_filter)
        dialog.set_current_name("export.png")
        response = dialog.run()
        filename = dialog.get_filename()
        dialog.destroy()
        if response != Gtk.ResponseType.OK or not filename:
            return
        if not filename.lower().endswith(".png"):
            filename += ".png"

        # Re-capture at the requested resolution rather than upscaling
        # the small preview image - see VismolGLCore.render_to_image().
        full_res_array = self.glwidget.capture_screenshot(scale_factor=scale_factor)
        if full_res_array is None:
            self.status_label.set_text("Failed to capture the scene in high resolution.")
            return

        full_res_img = Image.fromarray(full_res_array)
        preset = self._current_preset()
        final = preset.apply(full_res_img)
        final.save(filename)
        self.status_label.set_text("Exported: {} ({}x)".format(filename, scale_factor))

    def on_export(self, widget):
        scale_text = self.scale_combo.get_active_text()
        scale_factor = int(scale_text[0])

        dialog = Gtk.FileChooserDialog(
            title="Export PNG",
            parent=self,
            action=Gtk.FileChooserAction.SAVE
        )

        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_SAVE, Gtk.ResponseType.OK
        )

        file_filter = Gtk.FileFilter()
        file_filter.set_name("PNG image (*.png)")
        file_filter.add_pattern("*.png")
        dialog.add_filter(file_filter)

        # Reutiliza último diretório
        if hasattr(self, "last_export_dir") and self.last_export_dir:
            dialog.set_current_folder(self.last_export_dir)


        if getattr(self,'last_export_file', False):
            filename = self._get_unique_filename(self.last_export_file)
        else:
            filename = self._get_unique_filename("export.png")
        

        
        
        #dialog.set_current_name("export.png")
        dialog.set_current_name(filename)

        response = dialog.run()
        filename = dialog.get_filename()
        dialog.destroy()
        self.last_export_file = filename
        if response != Gtk.ResponseType.OK or not filename:
            return

        if not filename.lower().endswith(".png"):
            filename += ".png"

        # Salva diretório para próxima chamada
        self.last_export_dir = os.path.dirname(filename)
        
        # Gera nome único se já existir
        full_res_array = self.glwidget.capture_screenshot(
            scale_factor=scale_factor
        )

        if full_res_array is None:
            self.status_label.set_text(
                "Failed to capture the scene in high resolution."
            )
            return

        full_res_img = Image.fromarray(full_res_array)
        preset = self._current_preset()
        final = preset.apply(full_res_img)
        final.save(filename)

        self.status_label.set_text(
            "Exported: {} ({}x)".format(filename, scale_factor)
        )
        
    def _get_unique_filename(self, filepath):
        """
        Se filepath já existir, gera:
        export.png
        export_1.png
        export_2.png
        ...
        """
        if not os.path.exists(filepath):
            return filepath

        folder, filename = os.path.split(filepath)
        base, ext = os.path.splitext(filename)

        counter = 1
        while True:
            new_name = f"{base}_{counter}{ext}"
            new_path = os.path.join(folder, new_name)

            if not os.path.exists(new_path):
                return new_path

            counter += 1



# ---- USO ----
def open_preview(image_array, glwidget):
    win = PreviewWindow(image_array, glwidget)
    win.show_all()

#![no_std]
#![feature(bigint_helper_methods)]

extern crate alloc;

use core::num::NonZeroUsize;

use alloc::boxed::Box;
use alloc::vec::Vec;
use bitfield_struct::bitfield;

use nx::gpu::surface::Surface;
use nx::service::vi::LayerFlags;

use nx::gpu::{self, MultiFence};
use nx::result::*;
use nx::service::hid::{AnalogStickState, DebugPadButton, TouchState};

#[cfg(feature = "std")]
#[panic_handler]
fn panic_handler(info: &panic::PanicInfo) -> ! {
    util::simple_panic_handler::<LmLogger>(info, abort::AbortLevel::FatalThrow())
}

use helpers::PromotingMul;
mod helpers {
    pub(crate) trait PromotingMul {
        type Output;
        fn promoting_mul(&self, other: Self) -> Self::Output;
    }
    macro_rules! promoting_mul_impl {
        ($in:ty, $out:ty) => {
            impl PromotingMul for $in {
                type Output = $out;
                fn promoting_mul(&self, other: Self) -> Self::Output {
                    let (low, high) = self.widening_mul(other);
                    <$out>::from(high) << (core::mem::size_of::<$out>() / 8) | <$out>::from(low)
                }
            }
        };
    }

    promoting_mul_impl!(u8, u16);
    promoting_mul_impl!(u16, u32);
}

mod config {
    const SCREEN_WIDTH: usize = 1920;
    const SCREEN_HEIGHT: usize = 1080;

    /*
    extern u16 LayerWidth;                  ///< Width of the Tesla layer
    extern u16 LayerHeight;                 ///< Height of the Tesla layer
    extern u16 LayerPosX;                   ///< X position of the Tesla layer
    extern u16 LayerPosY;                   ///< Y position of the Tesla layer
    extern u16 FramebufferWidth;            ///< Width of the framebuffer
    extern u16 FramebufferHeight;           ///< Height of the framebuffer
    extern u64 launchCombo;                 ///< Overlay activation key combo
    */
}

/// One point rectangle
#[derive(Debug, Default, Clone, Copy)]
pub struct Rect {
    pub left: usize,
    pub top: usize,
    pub width: usize,
    pub height: usize,
}

impl Rect {
    const EMTPY: Rect = Rect {
        left: 0,
        top: 0,
        width: 0,
        height: 0,
    };
    const MAX: Rect = Rect {
        left: 0,
        top: 0,
        width: usize::MAX,
        height: usize::MAX,
    };

    pub fn right(&self) -> usize {
        self.left.saturating_add(self.width)
    }
    pub fn bottom(&self) -> usize {
        self.top.saturating_add(self.height)
    }

    #[inline]
    pub fn contains(&self, x: usize, y: usize) -> bool {
        (self.left..self.right()).contains(&x) && (self.top..self.bottom()).contains(&y)
    }

    pub const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    pub fn intersect(&self, other: Self) -> Self {
        let left = self.left.clamp(other.left, other.right());
        let top = self.top.clamp(other.top, other.bottom());
        let width = self.right().clamp(other.left, other.right()) - left;
        let height = self.bottom().clamp(other.top, other.bottom()) - top;
        Rect {
            left,
            top,
            width,
            height,
        }
    }
}

/// RGBA (4444) Color Representation
#[bitfield(u16)]
pub struct Color {
    /// Alpha channel
    #[bits(4)]
    a: u8,
    /// Blue channel
    #[bits(4)]
    b: u8,
    /// Green channel
    #[bits(4)]
    g: u8,
    /// Red channel
    #[bits(4)]
    r: u8,
}

impl Color {
    /// Default background color
    const BACKGROUND: Color = Self::from_values(0x0, 0x0, 0x0, 0xD);
    /// Transparent layer
    const TRANSPARENT: Color = Self::from_values(0x0, 0x0, 0x0, 0x0);
    /// Greenish highlight color
    const HIGHLIGHT: Color = Self::from_values(0x0, 0xF, 0xD, 0xF);
    /// Border color
    const FRAME: Color = Self::from_values(0x7, 0x7, 0x7, 0xF);
    const HANDLE: Color = Self::from_values(0x5, 0x5, 0x5, 0xF);
    const TEXT: Color = Self::from_values(0xF, 0xF, 0xF, 0xF);
    const DESCRIPTION: Color = Self::from_values(0xA, 0xA, 0xA, 0xF);
    const HEADER_BAR: Color = Self::from_values(0xC, 0xC, 0xC, 0xF);
    const CLICK_ANIMATION: Color = Self::from_values(0x0, 0x2, 0x2, 0xF);

    pub const fn as_rgba4444(&self) -> u16 {
        self.into_bits()
    }

    pub const fn as_abgr4444(&self) -> u16 {
        Self::from_values(self.a(), self.b(), self.g(), self.r()).into_bits()
    }

    pub const fn from_values(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self::new().with_r(r).with_g(g).with_b(b).with_a(a)
    }

    #[inline]
    pub fn blend_channel(left: u8, right: u8, alpha: u8) -> u8 {
        assert!(alpha <= 0xF, "Invalid alpha channel for 4-bit channel");

        let one_minus_alpha = 0xFF - alpha;

        // We can cast back to u8, as a division by 0xFF is the same as a shift right by 8.
        ((right.promoting_mul(alpha) + left.promoting_mul(one_minus_alpha)) / 0xFF) as u8
    }

    pub fn blend_with(&mut self, other: Self, blend_alpha: bool) {
        self.set_r(Self::blend_channel(self.a(), other.r(), self.a()));
        self.set_g(Self::blend_channel(self.g(), other.g(), self.a()));
        self.set_b(Self::blend_channel(self.b(), other.b(), self.a()));
        self.set_a(if blend_alpha {
            self.a() + other.a()
        } else {
            self.a()
        });
    }
}

/// Direction in which focus moved before landing on the currently focused element
/// Keeping compatibility from ambiguous left/right in original libtesla
#[repr(C)]
pub enum FocusDirection {
    /// Focus was placed on the element programatically without user input
    None,
    /// Focus moved upwards                
    Up,
    /// Focus moved downwards                   
    Down,
    /// Focus moved from left to right
    Left,
    /// Focus moved from right to left
    Right,
}

/// Current input control mode
#[repr(C)]
pub enum InputMode {
    /// Input from controller
    Controller,
    /// Touch input
    Touch,
    /// Moving/scrolling touch input
    TouchScroll,
}

///Combo key mapping
pub struct KeyInfo {
    key: DebugPadButton,
    name: &'static str,
    glyph: char,
}

impl KeyInfo {}

pub struct Renderer<'r> {
    pub opacity: f32,
    display_handle: nx::gpu::Context,
    pub(crate) surface: Surface,
    scisoring_config: Vec<Rect>,
    standard_font: Option<rusttype::Font<'r>>,
    local_font: Option<rusttype::Font<'r>>,
    external_font: Option<rusttype::Font<'r>>,
}

impl<'r> Renderer<'r> {
    pub fn new(x: f32, y: f32, width: u32, height: u32, opacity: f32) -> Result<Self> {
        //nx::hbl::AbiConfigEntry
        let mut gpu_ctx = gpu::Context::new(
            gpu::NvDrvServiceKind::Applet,
            gpu::ViServiceKind::Manager,
            0x800000,
        )?;
        let surface = gpu_ctx.create_managed_layer_surface(
            "Default",
            0,
            LayerFlags::Default(),
            x,
            y,
            width,
            height,
            gpu::LayerZ::Max,
            2,
            gpu::ColorFormat::R4G4B4A4,
            gpu::PixelFormat::RGBA_4444,
            gpu::Layout::BlockLinear,
        )?;

        Ok(Self {
            opacity,
            display_handle: gpu_ctx,
            surface,
            scisoring_config: Vec::new(),
            standard_font: None,
            local_font: None,
            external_font: None,
        })
    }

    fn wait_vsync_event(&mut self) {
        self.surface.wait_vsync_event(-1i64);
    }

    pub fn set_opacity(&mut self, opacity: f32) {
        self.opacity = opacity.clamp(0f32, 1f32);
    }

    fn opacity_pass(&self, color: &Color) -> Color {
        color
            .clone()
            .with_a({ color.a() as f32 * self.opacity } as u8)
    }

    fn get_framebuffer(&'r mut self) -> Result<FrameBuffer<'r>> {
        let (buffer, buffer_length, slot, fence_present, fences) =
            self.surface.dequeue_buffer(false)?;

        Ok(FrameBuffer {
            width: self.surface.get_width() as usize,
            height: self.surface.get_height() as usize,
            buffer: unsafe {
                core::slice::from_raw_parts_mut(
                    buffer as _,
                    buffer_length as usize / core::mem::size_of::<Color>(),
                )
            },
            context_ref: &*self,
            fence_present,
            fences,
            slot,
        })
    }
}

pub struct FrameBuffer<'b> {
    pub width: usize,
    pub height: usize,
    pub buffer: &'b mut [Color],
    pub context_ref: &'b Renderer<'b>,
    pub fence_present: bool,
    pub fences: MultiFence,
    pub slot: i32,
}

impl<'b> FrameBuffer<'b> {
    fn stride_item_count(&self) -> usize {
        self.context_ref.surface.compute_stride() as usize / core::mem::size_of::<Color>()
    }
    pub fn clear(&mut self) {
        self.draw_rect(
            Rect {
                left: 0,
                top: 0,
                width: self.width,
                height: self.height,
            },
            Color::BACKGROUND,
        );
    }

    pub fn draw_rect(&mut self, mut rect: Rect, color: Color) {
        if let Some(&scisoring_area) = self.context_ref.scisoring_config.last() {
            rect.intersect(scisoring_area);
        }

        if rect.is_empty() {
            return;
        }

        for x_pixel in rect.left..rect.right() {
            for y_pixel in rect.top..rect.bottom() {
                self.buffer[y_pixel * self.stride_item_count() + x_pixel].blend_with(color, true);
            }
        }
    }

    fn draw_box(&mut self, rect: Rect, line_width: usize, color: Color) {
        let line_offsets = core::cmp::max(1, line_width/2);
        //top
        self.draw_rect(
            Rect {
                left: rect.left - line_offsets,
                top: rect.top - line_offsets,
                width: rect.width + 2*line_offsets,
                height: 2*line_offsets,
            },
            color,
        );
        //bottom
        self.draw_rect(
            Rect {
                left: rect.left - line_offsets,
                top: rect.bottom() - line_offsets,
                width: rect.width + 2*line_offsets,
                height: 2*line_offsets,
            },
            color,
        );
        //left
        self.draw_rect(
            Rect {
                left: rect.left - line_offsets,
                top: rect.top - line_offsets,
                width: 2*line_offsets,
                height: rect.height + 2*line_offsets,
            },
            color,
        );
        //right
        self.draw_rect(
            Rect {
                left: rect.right() - line_offsets,
                top: rect.top - line_offsets,
                width: 2*line_offsets,
                height: rect.height + 2*line_offsets,
            },
            color,
        );
    }

}

pub mod elm {
    use nx::service::hid::NpadButton;

    use super::*;

    pub trait Element {
        fn request_focus(&mut self, focus_direction: FocusDirection) -> Option<&mut dyn Element> {None}

        fn on_click(&mut self, keys: u64) -> bool {
            return false;
        }

        fn on_touch(
            &mut self,
            touch_state: &TouchState,
            previous_touch_state: Option<&TouchState>,
        ) -> bool {
            return false;
        }

        /// Called every frame
        /// returns true if the input is consumed, else false.
        fn on_controller_input(
            &mut self,
            new_keys: u64,
            held_keys: u64,
            touch_state: Option<&TouchState>,
            joy_stick_pos_left: &AnalogStickState,
            joy_stick_pos_right: &AnalogStickState,
        ) -> bool {
            return false;
        }

        fn draw(&mut self, renderer: &mut FrameBuffer);

        fn bounds_rect(&self) -> Rect;

        fn draw_background(&self, framebuffer: &mut FrameBuffer, color: Option<Color>) {
            let Rect {
                left,
                top,
                width,
                height,
            } = self.bounds_rect();
            framebuffer.draw_rect(self.bounds_rect(), color.unwrap_or(Color::BACKGROUND));
        }

        fn draw_highlight(&self, framebuffer: &mut FrameBuffer, color: Option<Color>) {
            let Rect {
                left, top, width, ..
            } = self.bounds_rect();
            let color = color.unwrap_or(Color::HIGHLIGHT);

            framebuffer.draw_rect(
                Rect {
                    left: left - 4,
                    top: top - 4,
                    width: width + 8,
                    height: 4,
                },
                color,
            );
        }

        fn trigger_highlight_shake(&mut self, direction: FocusDirection) {}

        fn trigger_click_animation(&mut self, input: InputMode) {}
        fn reset_click_animation(&mut self) {}
        fn draw_click_animation(&mut self) {}

        #[inline]
        fn in_bounds(&self, x: usize, y: usize) {
            self.bounds_rect().contains(x, y);
        }

        fn set_parent(&mut self, parent: alloc::boxed::Box<dyn Element>);

        fn get_parent(&self) -> Option<&Box<dyn Element>>;
    }

    struct DebugRectangle {
        bounds: Rect,
        parent: Option<Box<dyn Element>>,
        color: Color

    }

    impl DebugRectangle {
        pub fn new(color: Color, bounds: Rect) -> Self {
            Self { bounds, parent: None, color }
        }
    }

    impl Element for DebugRectangle {
        fn get_parent(&self) -> Option<&Box<dyn Element>> {
            self.parent.as_ref()
        }

        fn set_parent(&mut self, parent: alloc::boxed::Box<dyn Element>) {
            self.parent = Some(parent)
        }

        fn bounds_rect(&self) -> Rect {
            self.bounds
        }

        fn draw(&mut self, renderer: &mut FrameBuffer) {
            renderer.draw_rect(self.bounds, self.color);
        }
    }

    struct TrackBar {
        bounds: Rect,
        parent: Option<Box<dyn Element>>,
        color: Color,
        value: u8,
        icon: char,
        value_changed_callback: Option<fn(u8)>
        touch_locked: bool
    }

    impl TrackBar {
        pub fn new(bounds: Rect, color: Color, value: Option<u8>, icon: char, value_changed_callback: Option< fn(u8)>, touch_locked: bool) -> Self {
            Self { bounds, parent: None, color, value: value.unwrap_or(0), icon , value_changed_callback, touch_locked}
        }
    }

    impl Element for TrackBar {
        fn request_focus(&mut self, focus_direction: FocusDirection) -> Option<&mut dyn Element> {
            Some(self)
        }

        fn get_parent(&self) -> Option<&Box<dyn Element>> {
            self.parent.as_ref()
        }

        fn draw(&mut self, renderer: &mut FrameBuffer) {
            
        }

        fn set_parent(&mut self, parent: alloc::boxed::Box<dyn Element>) {
            self.parent = Some(parent);
        }

        fn bounds_rect(&self) -> Rect {
            self.bounds
        }

        fn on_controller_input(
                    &mut self,
                    _new_keys: u64,
                    held_keys: u64,
                    _touch_state: Option<&TouchState>,
                    _joy_stick_pos_left: &AnalogStickState,
                    _joy_stick_pos_right: &AnalogStickState,
                ) -> bool {
            static ANY_LEFT: u64 = NpadButton::Left().get() | NpadButton::StickLLeft().get() | NpadButton::StickRLeft().get();
            static ANY_RIGHT: u64 = NpadButton::Right().get() | NpadButton::StickLRight().get() | NpadButton::StickRRight().get();
            match (held_keys & ANY_LEFT != 0, held_keys & ANY_RIGHT != 0) {
                (true, true) | (false, false) => true,
                (true, false) if self.value < 100 => {
                    self.value += 1;
                    if let Some(callback) = self.value_changed_callback {
                        callback(self.value);
                    }
                    true
                },
                (false, true) if self.value > 0 => {
                    self.value -= 1;
                    if let Some(callback) = self.value_changed_callback {
                        callback(self.value);
                    }
                    true
                },
                (_,_) => false
            }
        }

        
        fn draw_background(&self, framebuffer: &mut FrameBuffer, color: Option<Color>) {
            todo!()
        }
    }
}

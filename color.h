#ifndef COLOR_IN_TERMINAL
#define COLOR_IN_TERMINAL

#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>

enum colorof {
    red, white, blue, green, magenta, grey, yellow, cyan, black, orange, pink, brown,
    lavender, navy, teal, coral, gold, darkgreen, mediumslateblue, tomato, salmon,
    khaki, peru, plum, turquoise, violet, wheat, sienna, slateblue, olive, mintcream,
    midnightblue, maroon, lime, indigo, ivory, hotpink, firebrick, fuchsia, forestgreen,
    gainsboro, ghostwhite, goldenrod, gray, greenyellow, honeydew, indianred, ivoryblack,
    lavenderblush, lemonchiffon, lightblue, lightcoral, lightcyan, lightgoldenrodyellow,
    lightgray, lightgreen, lightpink, lightsalmon, lightseagreen, lightskyblue,
    lightslategray, lightsteelblue, lightyellow, limegreen, linen, magentared,
    mediumaquamarine, mediumblue, mediumorchid, mediumpurple, mediumseagreen,
    mediumspringgreen, mediumturquoise, mediumvioletred, midnightblue2, mintcream2,
    mistyrose, moccasin, navajowhite, oldlace, olivedrab, orangered, orchid,
    palegoldenrod, palegreen, paleturquoise, palevioletred, papayawhip, peachpuff,
    peru2, powderblue, rosybrown, royalblue, saddlebrown, salmon2, sandybrown, seashell,
    sienna2, skyblue, slategray, snow, springgreen, steelblue, tanin, thistle, tomato2,
    turquoise2, violet2, wheat2, whitesmoke, yellowgreen
};

struct RGB {
    int r, g, b;
};

static const RGB color_table[111] = {
    {255, 0, 0}, {255, 255, 255}, {0, 0, 255}, {0, 255, 0}, {255, 0, 255}, {128, 128, 128},
    {255, 255, 0}, {0, 255, 255}, {0, 0, 0}, {255, 165, 0}, {255, 192, 203}, {165, 42, 42},
    {230, 230, 250}, {0, 0, 128}, {0, 128, 128}, {255, 127, 80}, {255, 215, 0}, {0, 100, 0},
    {123, 104, 238}, {255, 99, 71}, {250, 128, 114}, {240, 230, 140}, {205, 133, 63},
    {221, 160, 221}, {64, 224, 208}, {238, 130, 238}, {245, 222, 179}, {160, 82, 45},
    {106, 90, 205}, {128, 128, 0}, {245, 255, 250}, {25, 25, 112}, {128, 0, 0}, {0, 255, 0},
    {75, 0, 130}, {255, 255, 240}, {255, 105, 180}, {178, 34, 34}, {255, 0, 255}, {34, 139, 34},
    {220, 220, 220}, {248, 248, 255}, {218, 165, 32}, {128, 128, 128}, {173, 255, 47},
    {240, 255, 240}, {205, 92, 92}, {0, 0, 0}, {255, 240, 245}, {255, 250, 205}, {173, 216, 230},
    {240, 128, 128}, {224, 255, 255}, {250, 250, 210}, {211, 211, 211}, {144, 238, 144},
    {255, 182, 193}, {255, 160, 122}, {32, 178, 170}, {135, 206, 250}, {119, 136, 153},
    {176, 196, 222}, {255, 255, 224}, {50, 205, 50}, {250, 240, 230}, {255, 0, 144},
    {102, 205, 170}, {0, 0, 205}, {186, 85, 211}, {147, 112, 219}, {60, 179, 113},
    {0, 250, 154}, {72, 209, 204}, {199, 21, 133}, {25, 25, 112}, {245, 255, 250},
    {255, 228, 225}, {255, 228, 181}, {255, 222, 173}, {253, 245, 230}, {107, 142, 35},
    {255, 69, 0}, {218, 112, 214}, {238, 232, 170}, {152, 251, 152}, {175, 238, 238},
    {219, 112, 147}, {255, 239, 213}, {255, 218, 185}, {205, 133, 63}, {176, 224, 230},
    {188, 143, 143}, {65, 105, 225}, {139, 69, 19}, {250, 128, 114}, {244, 164, 96},
    {255, 245, 238}, {160, 82, 45}, {135, 206, 235}, {112, 128, 144}, {255, 250, 250},
    {0, 255, 127}, {70, 130, 180}, {210, 180, 140}, {216, 191, 216}, {255, 99, 71},
    {64, 224, 208}, {238, 130, 238}, {245, 222, 179}, {245, 245, 245}, {154, 205, 50}
};

namespace terminal_color {

    // Hidden internal namespace so helpers don't pollute your global scope
    namespace internal {
        inline RGB get_rgb(colorof c) {
            if (c >= 0 && c < 111) return color_table[c];
            return {0, 0, 0};
        }

        inline int char_to_int(char a) {
            if (a >= '0' && a <= '9') return a - '0';
            if (a >= 'a' && a <= 'f') return a - 'a' + 10;
            if (a >= 'A' && a <= 'F') return a - 'A' + 10;
            return -1;
        }

        inline int hex_to_val(const std::string& a) {
            if (a.length() == 2) {
                return char_to_int(a[0]) * 16 + char_to_int(a[1]);
            }
            return -1;
        }

        inline RGB get_rgb(const std::string& colorStr) {
            if (colorStr.empty()) return {0,0,0};
            if (colorStr[0] == '#') {
                return {
                    hex_to_val(colorStr.substr(1, 2)),
                    hex_to_val(colorStr.substr(3, 2)),
                    hex_to_val(colorStr.substr(5, 2))
                };
            } 
            else if (colorStr.substr(0, 4) == "rgb(") {
                int endPos = colorStr.find(')', 4);
                std::string rgbContent = colorStr.substr(4, endPos - 4);
                size_t pos1 = rgbContent.find(',');
                size_t pos2 = rgbContent.find(',', pos1 + 1);
                return {
                    std::stoi(rgbContent.substr(0, pos1)),
                    std::stoi(rgbContent.substr(pos1 + 1, pos2 - pos1 - 1)),
                    std::stoi(rgbContent.substr(pos2 + 1))
                };
            }
            throw std::invalid_argument("Invalid color format: " + colorStr);
        }

        // ONE unified logic function to prevent repeating the stringstream 6 times
        template <class T>
        inline std::string build_ansi(RGB fg, RGB bg, bool use_bg, const T& value, 
                                      bool bold, bool underline, bool italic, 
                                      bool reversed, bool strikethrough) {
            std::stringstream ss;
            ss << "\033[";
            if (bold) ss << "1;";
            if (italic) ss << "3;";
            if (underline) ss << "4;";
            if (reversed) ss << "7;";
            if (strikethrough) ss << "9;";

            if (use_bg) {
                ss << "48;2;" << bg.r << ";" << bg.g << ";" << bg.b << ";";
            }
            
            ss << "38;2;" << fg.r << ";" << fg.g << ";" << fg.b << "m" << value << "\033[0m";
            return ss.str();
        }
    }

    // ==========================================
    // 1. FOREGROUND ONLY (No background required)
    // ==========================================
    template <class beeta>
    std::string color(std::string fg, beeta value, bool bold = false, bool underline = false, bool italic = false, bool reversed = false, bool strikethrough = false) {
        return internal::build_ansi(internal::get_rgb(fg), {0,0,0}, false, value, bold, underline, italic, reversed, strikethrough);
    }

    template <class beeta>
    std::string color(colorof fg, beeta value, bool bold = false, bool underline = false, bool italic = false, bool reversed = false, bool strikethrough = false) {
        return internal::build_ansi(internal::get_rgb(fg), {0,0,0}, false, value, bold, underline, italic, reversed, strikethrough);
    }

    // ==========================================
    // 2. FOREGROUND + STRING BACKGROUND
    // ==========================================
    template <class beeta>
    std::string color(std::string fg, beeta value, std::string bg, bool bold = false, bool underline = false, bool italic = false, bool reversed = false, bool strikethrough = false) {
        return internal::build_ansi(internal::get_rgb(fg), internal::get_rgb(bg), true, value, bold, underline, italic, reversed, strikethrough);
    }

    template <class beeta>
    std::string color(colorof fg, beeta value, std::string bg, bool bold = false, bool underline = false, bool italic = false, bool reversed = false, bool strikethrough = false) {
        return internal::build_ansi(internal::get_rgb(fg), internal::get_rgb(bg), true, value, bold, underline, italic, reversed, strikethrough);
    }

    // ==========================================
    // 3. FOREGROUND + ENUM BACKGROUND
    // ==========================================
    template <class beeta>
    std::string color(std::string fg, beeta value, colorof bg, bool bold = false, bool underline = false, bool italic = false, bool reversed = false, bool strikethrough = false) {
        return internal::build_ansi(internal::get_rgb(fg), internal::get_rgb(bg), true, value, bold, underline, italic, reversed, strikethrough);
    }

    template <class beeta>
    std::string color(colorof fg, beeta value, colorof bg, bool bold = false, bool underline = false, bool italic = false, bool reversed = false, bool strikethrough = false) {
        return internal::build_ansi(internal::get_rgb(fg), internal::get_rgb(bg), true, value, bold, underline, italic, reversed, strikethrough);
    }

}
#endif
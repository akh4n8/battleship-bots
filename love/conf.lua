-- love/conf.lua
-- Love2D Window & Runtime Configuration for Battleship AI Research Lab Visualizer

function love.conf(t)
    t.identity = "battleship_bots_lab"
    t.version = "11.4"
    t.console = false

    t.window.title = "Battleship AI Research Lab — Live Play & Forensic Diagnostics (LuaJIT + Love2D)"
    t.window.width = 1240
    t.window.height = 760
    t.window.resizable = true
    t.window.minwidth = 1100
    t.window.minheight = 680
    t.window.vsync = 1
    t.window.msaa = 4
    t.window.highdpi = true

    t.modules.audio = false
    t.modules.joystick = false
    t.modules.physics = false
    t.modules.touch = false
    t.modules.video = false
end

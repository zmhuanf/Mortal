//! akagi_mjai_bot：stdin/stdout 桥接进程，把 libriichi（Python）的 mjai 事件流喂给
//! `native_bot::Engine` 并回传动作，让 Rust 小模型能作为 Python 引擎参与竞技场
//!
//! 一个进程服务一组局（player_id_idx 分片），每局一个 Engine（seat 固定）
//!
//! stdin 每行一个 JSON 命令：
//!   {"__cmd":"new_game","game":N,"seat":S}   新局，创建该局 Engine
//!   {"__cmd":"frame","game":N,"events":"[...]","full":bool}
//!     events 为 mjai 事件数组 JSON；full=true 时按已喂计数去重，否则视为增量
//!     处理完毕立即输出一行动作（mjai 事件 JSON，空行为无动作）
//!   {"__cmd":"drop_game","game":N}           释放该局状态
//!
//! 输出必须逐帧 flush，Python 端 readline 依赖其即时性

use std::collections::HashMap;
use std::io::{self, BufRead, BufWriter, Write};

use native_bot::defaults::WEIGHTS_4P;
use native_bot::engine::{BotAction, Engine};
use riichienv_core::replay::MjaiEvent;
use serde_json::{json, Value};

struct GameSlot {
    engine: Engine,
    fed: usize,
}

fn main() -> anyhow::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let mut games: HashMap<u64, GameSlot> = HashMap::new();

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let Ok(cmd) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        match cmd.get("__cmd").and_then(Value::as_str) {
            Some("new_game") => {
                let game = cmd["game"].as_u64().unwrap_or(0);
                let seat = cmd["seat"].as_u64().unwrap_or(0) as u8;
                let engine = Engine::new(WEIGHTS_4P.to_vec(), 4, seat)?;
                games.insert(game, GameSlot { engine, fed: 0 });
            }
            Some("frame") => {
                let game = cmd["game"].as_u64().unwrap_or(0);
                let Some(slot) = games.get_mut(&game) else {
                    writeln!(out)?;
                    out.flush()?;
                    continue;
                };
                let events = cmd["events"].as_str().unwrap_or("[]");
                let full = cmd.get("full").and_then(Value::as_bool).unwrap_or(false);
                let evs: Vec<Value> = serde_json::from_str(events).unwrap_or_default();
                if full {
                    for ev in evs.iter().skip(slot.fed) {
                        feed_ev(&mut slot.engine, ev);
                    }
                    slot.fed = evs.len();
                } else {
                    for ev in &evs {
                        feed_ev(&mut slot.engine, ev);
                    }
                    slot.fed += evs.len();
                }
                match slot.engine.decide().ok().flatten() {
                    Some(d) => match bot_action_to_json(&d.action, slot.engine.seat()) {
                        Some(s) => writeln!(out, "{s}")?,
                        // Pass：libriichi 以 `none` 事件表示无反应
                        None => writeln!(out, r#"{{"type":"none"}}"#)?,
                    },
                    None => writeln!(out, r#"{{"type":"none"}}"#)?,
                }
                out.flush()?;
            }
            Some("reset") => {
                // ctx.log 是小局级，跨小局事件计数必须归零，否则 full 去重会跳过 start_kyoku
                if let Some(slot) = games.get_mut(&cmd["game"].as_u64().unwrap_or(0)) {
                    slot.fed = 0;
                }
            }
            Some("drop_game") => {
                games.remove(&cmd["game"].as_u64().unwrap_or(0));
            }
            _ => {}
        }
    }
    Ok(())
}

fn feed_ev(engine: &mut Engine, ev: &Value) {
    if let Ok(ev) = serde_json::from_value::<MjaiEvent>(ev.clone()) {
        engine.feed(ev);
    }
}

fn bot_action_to_json(a: &BotAction, seat: u8) -> Option<String> {
    let v = match a {
        BotAction::Dahai { pai, tsumogiri } => {
            json!({"type":"dahai","actor":seat,"pai":pai,"tsumogiri":tsumogiri})
        }
        // libriichi 的 reach 事件无 pai 字段，打牌由 reach_accepted 后的下一次 decide 完成
        BotAction::Reach { .. } => json!({"type":"reach","actor":seat}),
        BotAction::Pon { target, pai, consumed } => {
            json!({"type":"pon","actor":seat,"target":target,"pai":pai,"consumed":consumed})
        }
        BotAction::Chi { target, pai, consumed } => {
            json!({"type":"chi","actor":seat,"target":target,"pai":pai,"consumed":consumed})
        }
        BotAction::Daiminkan { target, pai, consumed } => json!({
            "type":"daiminkan","actor":seat,"target":target,"pai":pai,"consumed":consumed
        }),
        BotAction::Ankan { consumed } => {
            json!({"type":"ankan","actor":seat,"consumed":consumed})
        }
        BotAction::Kakan { pai, consumed } => {
            json!({"type":"kakan","actor":seat,"pai":pai,"consumed":consumed})
        }
        BotAction::Hora { target } => json!({"type":"hora","actor":seat,"target":target}),
        BotAction::Kyushu => json!({"type":"ryukyoku"}),
        BotAction::Kita => json!({"type":"kita","actor":seat}),
        BotAction::Pass => return None,
    };
    Some(v.to_string())
}

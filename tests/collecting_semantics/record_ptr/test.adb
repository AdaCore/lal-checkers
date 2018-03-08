procedure Access_Test is
   type Point is record
      x : Integer;
      y : Integer;
   end record;
   type Rectangle is record
      top_left : access Point;
      bot_right : access Point;
   end record;

   p : Point;
   r : Rectangle := (p'Access, p'Access);
begin
   if r.top_left.all.x = 1 then
      r.bot_right.all.y := 2;
   else
      p := (1, 2);
   end if;
end Access_Test;

procedure Ex1 is
   type Point is record
      x : Integer;
      y : Integer;
   end record;
   type Rectangle is record
      top_left : access Point;
      bot_right : access Point;
   end record;

   p : Point := (1, 2);
   r : Rectangle := (p'Access, p'Access);
begin
   r.top_left.all.x := 3;
end Ex1;

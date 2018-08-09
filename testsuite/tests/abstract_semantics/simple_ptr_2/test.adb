procedure Test is
   type Point is record
      x : Integer;
      y : Integer;
   end record;
   x : access Integer;
   p : Point := (1, 2);
   b : Boolean;
begin
   x := p.x'Access;
   p.x := 3;
   p.y := x.all;
end Ex1;

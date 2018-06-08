procedure Ex1 is
   type Point is record
      x : Integer;
      y : Integer;
   end record;
   x : access Integer;
   p : Point;
   b : Boolean;
begin
   x := p.x'Access;
end Ex1;
